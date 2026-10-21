# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn

from kornia.core.download import load_state_dict_from_url
from kornia.core.external import PILImage as Image
from kornia.core.external import onnx
from kornia.core.mixin.onnx import ONNXExportMixin, _preserved_rng, _returns_list
from kornia.models.base import ModelBase
from kornia.models.processors import OutputRangePostProcessor, ResizePreProcessor
from kornia.models.rrdbnet import RRDBNet
from kornia.models.small_sr import SmallSRNetWrapper

__all__ = ["RRDBNetBuilder", "SmallSRBuilder", "SuperResolution", "SuperResolutionConfig"]

# `SmallSRBuilder`'s wrapper resizes every input to this before the network, so the
# pipeline's output side is always this times `upscale_factor`, whatever `image_size` is.
_SMALL_SR_INPUT_SIZE = 224

_URLs = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}


# TODO: support patching -> SR -> unpatching pipeline
@dataclass
class SuperResolutionConfig:
    """Configuration to construct a :class:`SuperResolution` model.

    The fields mirror the arguments accepted by :class:`RRDBNetBuilder` and
    :class:`SmallSRBuilder`, so a config can drive either builder.

    Args:
        model_name: Name of the model variant to build. RRDB variants are
            ``"RealESRGAN_x4plus"``, ``"RealESRNet_x4plus"``,
            ``"RealESRGAN_x4plus_anime_6B"`` and ``"RealESRGAN_x2plus"``;
            the lightweight variant is ``"small_sr"``.
        pretrained: If ``True``, load pretrained weights for the selected variant.
        upscale_factor: Integer scale used by the lightweight variant. Ignored by
            the RRDB variants, whose scale is fixed by ``model_name``. The lightweight
            variant resizes every input to 224 x 224 first, discarding its aspect ratio,
            so its output is ``224 * upscale_factor`` square (672 x 672 for 3, from a
            100 x 50 input as from a 480 x 640 one). Its pretrained weights exist only
            for 3: another factor with ``pretrained=True`` raises a size-mismatch
            :class:`RuntimeError`.
        image_size: Optional fixed input size for the lightweight variant. It declares
            the size fed to the wrapper; it does not change the pre-processor, which
            always resizes to 224, so the output side stays ``224 * upscale_factor``.

    """

    model_name: str = "small_sr"
    pretrained: bool = True
    upscale_factor: int = 3
    image_size: Optional[int] = None


class SuperResolution(ModelBase[SuperResolutionConfig], ONNXExportMixin):
    """SuperResolution is a module that wraps an super resolution model."""

    name: str = "super_resolution"
    input_image_size: Optional[int]
    output_image_size: Optional[int]
    pseudo_image_size: Optional[int]

    def __init__(
        self,
        model: nn.Module,
        pre_processor: nn.Module,
        post_processor: nn.Module,
        name: Optional[str] = None,
    ) -> None:
        """Initialize SuperResolution.

        Args:
            model: The super-resolution network.
            pre_processor: Pre-processing module applied to the input images.
            post_processor: Post-processing module applied to the network output.
            name: Optional name, used by :meth:`save` for file names.

        """
        super().__init__()
        self.model = model.eval()
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        if name is not None:
            self.name = name
        # `to_onnx` reads these three; builders overwrite the ones they know.
        # Without defaults they are annotations only, and export raises
        # AttributeError.
        self.input_image_size = None
        self.output_image_size = None
        self.pseudo_image_size = None

    @staticmethod
    def from_config(config: SuperResolutionConfig) -> "SuperResolution":
        """Build a super-resolution model from a configuration object.

        Dispatches to :class:`SmallSRBuilder` for the lightweight variant and to
        :class:`RRDBNetBuilder` for the RRDB variants.

        Args:
            config: Super-resolution configuration. See :class:`SuperResolutionConfig`.

        Returns:
            A :class:`SuperResolution` wrapper ready for inference.

        Raises:
            ValueError: If ``config.model_name`` is not a supported variant.

        """
        if config.model_name.lower() == "small_sr":
            return SmallSRBuilder.build(
                model_name=config.model_name,
                pretrained=config.pretrained,
                upscale_factor=config.upscale_factor,
                image_size=config.image_size,
            )
        if config.model_name not in _URLs:
            raise ValueError(
                f"Model {config.model_name} not found. Please choose from 'small_sr', "
                "'RealESRGAN_x4plus', 'RealESRNet_x4plus', 'RealESRGAN_x4plus_anime_6B', "
                "'RealESRGAN_x2plus'."
            )
        return RRDBNetBuilder.build(model_name=config.model_name, pretrained=config.pretrained)

    @torch.inference_mode()
    def forward(self, images: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass of the super resolution model.

        Args:
            images: If list of RGB images. Each image is a torch.Tensor with shape :math:`(3, H, W)`.
                If torch.Tensor, a torch.Tensor with shape :math:`(B, 3, H, W)`.

        Returns:
            output torch.Tensor.

        """
        output = self.pre_processor(images)
        # ``ResizePreProcessor`` returns an ``(images, sizes)`` tuple; a list of images goes through one at a time
        images = output[0] if isinstance(output, tuple) else output
        if isinstance(images, list):
            out_images = [self.model(image[None])[0] for image in images]
        else:
            out_images = self.model(images)
        return self.post_processor(out_images)

    def visualize(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        edge_maps: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        output_type: str = "torch",
    ) -> Union[torch.Tensor, List[torch.Tensor], List["Image.Image"]]:
        """Draw the super resolution results.

        Args:
            images: input torch.Tensor.
            edge_maps: super-resolved images; computed with :meth:`forward` when None.
            output_type: type of the output.

        Returns:
            output torch.Tensor.

        """
        if edge_maps is None:
            edge_maps = self.forward(images)
        output = []
        for edge_map in edge_maps:
            output.append(edge_map)

        return self._tensor_to_type(output, output_type, is_batch=isinstance(images, torch.Tensor))

    def save(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        edge_maps: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        directory: Optional[str] = None,
        output_type: str = "torch",
    ) -> None:
        """Save the super resolution results.

        Args:
            images: input torch.Tensor.
            edge_maps: super-resolved images; computed with :meth:`forward` when None.
            directory: where to save outputs.
            output_type: "torch" or "pil", passed to :meth:`visualize`; the files hold the same pixels either way.

        """
        outputs = self.visualize(images, edge_maps, output_type)
        self._save_outputs(images, directory, suffix="_src")
        self._save_outputs(outputs, directory, suffix="_sr")

    def to_onnx(
        self,
        onnx_name: Optional[str] = None,
        include_pre_and_post_processor: bool = True,
        save: bool = True,
        additional_metadata: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> "onnx.ModelProto":
        """Export the current super resolution model to an ONNX model file.

        Args:
            onnx_name:
                The name of the output ONNX file. If not provided, ``"kornia_<name>.onnx"``, with the model's
                ``name``, for example ``"kornia_small_sr.onnx"``.
            include_pre_and_post_processor:
                Whether to include the pre-processor and post-processor in the exported model.
            save:
                If True, write the model to ``onnx_name``; it is returned either way.
            additional_metadata:
                Additional metadata to add to the ONNX model.
            kwargs: Additional keyword arguments of :func:`torch.onnx.export`.

        Raises:
            ValueError: With the pre- and post-processor, if the output is a list with one image per input image and
                a ``dynamic_axes`` or ``dynamic_shapes`` argument makes the exported batch dynamic.

        Note:
            With the pre- and post-processor, an output that is a list with one image per input image is exported
            for a batch of one image: the exporters turn a list into one graph output per traced image, so a dynamic
            batch would return the traced batch's images at any batch size. The list comes from a post-processor that
            returns one, from the network itself, or from a pre-processor that returns a list of images, which
            :meth:`forward` runs through the network one at a time. Known tensor-output networks (the builders'
            RRDB and SmallSR networks, ``nn.Identity`` and ``nn.Conv2d``) with ``nn.Identity`` or ``ResizePreProcessor``
            before them and ``nn.Identity`` or ``OutputRangePostProcessor`` after them keep a dynamic batch without a
            probe, provided the wrapper and each component retain their known ``forward`` implementations and have
            no forward hooks. Other combinations are probed on one image at the export size in eval mode, without
            gradients and outside the export. A list result requires a fixed batch; a tensor result is checked again
            on the actual tracing batch before accepting a dynamic graph. A dynamic graph with per-image outputs
            raises a ``ValueError`` after the export, before anything is saved.

        """
        if onnx_name is None:
            onnx_name = f"kornia_{self.name}.onnx"

        # The dummy input must agree with a fixed input size, which ``to_onnx`` checks.
        pseudo_size = self.pseudo_image_size or self.input_image_size or 352
        pseudo_shape = [1, 3, pseudo_size, pseudo_size]
        # Exported for a dynamic batch, a list of per-image outputs comes out as the traced batch's images, or the
        # export fails: such a model is exported for one image.
        returns_list = None
        if include_pre_and_post_processor and not self._returns_one_tensor_without_a_probe():
            # The probe's input draws from the global generator, as the export's dummy does, and gives back what it
            # drew.
            with _preserved_rng():
                returns_list = _returns_list(self, torch.rand(pseudo_shape))
        batch = 1 if returns_list else -1

        with self._onnx_output_kind_kept(returns_list):
            return ONNXExportMixin.to_onnx(
                self,
                onnx_name,
                input_shape=[batch, 3, self.input_image_size or -1, self.input_image_size or -1],
                output_shape=[batch, 3, self.output_image_size or -1, self.output_image_size or -1],
                pseudo_shape=pseudo_shape,
                model=self if include_pre_and_post_processor else self.model,
                save=save,
                additional_metadata=additional_metadata,
                **kwargs,
            )

    def _returns_one_tensor_without_a_probe(self) -> bool:
        """Whether the wrapper, network and processors have known tensor-batch forward implementations.

        Unknown networks and overridden forwards need a probe even with the builders' processors. The builders'
        RRDB and SmallSR networks avoid an expensive extra forward when all implementations are unchanged.
        """
        forward = getattr(self.forward, "__func__", None)
        network = getattr(self.model.forward, "__func__", None)
        pre = getattr(self.pre_processor.forward, "__func__", None)
        post = getattr(self.post_processor.forward, "__func__", None)
        return (
            not any(module._forward_hooks or module._forward_pre_hooks for module in self.modules())
            and forward is SuperResolution.forward
            and network in (nn.Identity.forward, nn.Conv2d.forward, RRDBNet.forward, SmallSRNetWrapper.forward)
            and pre in (nn.Identity.forward, ResizePreProcessor.forward)
            and post in (nn.Identity.forward, OutputRangePostProcessor.forward)
        )

    def _onnx_fixed_batch_hint(self, module: Any, dummy_input: Any) -> Optional[str]:
        if module is not self or self._returns_one_tensor_without_a_probe() or not self._onnx_returns_list(dummy_input):
            return None
        return (
            "Use a network that returns one tensor for the batch, with a pre-processor that hands it one tensor and "
            "a post-processor that keeps it one, such as nn.Identity or ResizePreProcessor and "
            "OutputRangePostProcessor, to export a dynamic batch."
        )


class RRDBNetBuilder:
    """Provide a static factory to build Residual-in-Residual Dense Block (RRDB) networks."""

    @staticmethod
    def build(model_name: str = "RealESRNet_x4plus", pretrained: bool = True) -> SuperResolution:
        """Build a preconfigured RRDB (Residual-in-Residual Dense Block) model.

        The returned object is a :class:`SuperResolution` wrapper configured for
        inference. Internally, this factory selects architecture hyperparameters
        based on ``model_name``, and optionally loads pretrained weights.

        Args:
            model_name: Name of the RRDB variant to construct. Supported options are:
                ``"RealESRGAN_x4plus"``, ``"RealESRNet_x4plus"``,
                ``"RealESRGAN_x4plus_anime_6B"``, and ``"RealESRGAN_x2plus"``.
            pretrained: If ``True``, download and load pretrained weights for the
                selected model variant.

        Returns:
            A :class:`SuperResolution` wrapper ready for inference.

        Raises:
            ValueError: If ``model_name`` is not one of the supported variants.
        """
        if model_name in ("RealESRGAN_x4plus", "RealESRNet_x4plus"):
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        elif model_name == "RealESRGAN_x4plus_anime_6B":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        elif model_name == "RealESRGAN_x2plus":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        else:
            raise ValueError(
                f"Model {model_name} not found. Please choose from "
                "'RealESRGAN_x4plus', 'RealESRNet_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealESRGAN_x2plus'."
            )

        if pretrained:
            # The torch hub cache, as for every other kornia checkpoint. Real-ESRGAN's RealESRGANer prefers the EMA
            # weights and falls back to the plain ones.
            checkpoint = load_state_dict_from_url(_URLs[model_name], map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["params_ema" if "params_ema" in checkpoint else "params"], strict=True)
        model.eval()

        return SuperResolution(
            model,
            pre_processor=nn.Identity(),
            post_processor=OutputRangePostProcessor(min_val=0.0, max_val=1.0),
            name=model_name,
        )


class SmallSRBuilder:
    """Provide a static factory to build lightweight Super-Resolution models."""

    @staticmethod
    def build(
        model_name: str = "small_sr", pretrained: bool = True, upscale_factor: int = 3, image_size: Optional[int] = None
    ) -> SuperResolution:
        """Build a lightweight super-resolution (SR) model wrapper.

        SR stands for *super-resolution*, which means generating a higher
        resolution output from a lower resolution input.

        Args:
            model_name: Name of the model variant to build. Currently only
                ``"small_sr"`` is supported.
            pretrained: If ``True``, load pretrained weights for the selected model.
            upscale_factor: Integer scale used by ``SmallSRNetWrapper``. The pre-processor
                first resizes every input to 224 x 224, discarding its aspect ratio, so the
                output is ``224 * upscale_factor`` square: with ``3``, a 100 x 50 and a
                480 x 640 input both give 672 x 672. Pretrained weights exist only for
                ``3``; another factor with ``pretrained=True`` raises a size-mismatch
                :class:`RuntimeError`.
            image_size: Optional fixed input size. It declares the size fed to the
                wrapper and does not change the pre-processor, which always resizes to
                224; the output side is therefore ``224 * upscale_factor``.

        Returns:
            A :class:`SuperResolution` instance configured with resizing pre-processing
            and identity post-processing.

        Raises:
            ValueError: If ``model_name`` is not supported.
        """
        if model_name.lower() == "small_sr":
            model = SmallSRNetWrapper(upscale_factor, pretrained=pretrained)
        else:
            raise ValueError(f"Model {model_name} not found. Please choose from 'small_sr'.")

        sr = SuperResolution(
            model,
            pre_processor=ResizePreProcessor(_SMALL_SR_INPUT_SIZE, _SMALL_SR_INPUT_SIZE),
            post_processor=nn.Identity(),
            name=model_name,
        )
        # The pre-processor resizes to `_SMALL_SR_INPUT_SIZE` regardless of `image_size`,
        # so the wrapper's output side does not depend on the input size.
        sr.output_image_size = _SMALL_SR_INPUT_SIZE * upscale_factor
        if image_size is None:
            sr.pseudo_image_size = _SMALL_SR_INPUT_SIZE
        else:
            sr.input_image_size = image_size
            sr.pseudo_image_size = image_size
        return sr
