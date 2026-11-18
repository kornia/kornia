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

"""Upscale RGB images with pretrained super-resolution networks: Real-ESRGAN generators or a small sub-pixel network.

The builders' models return RGB images clamped to :math:`[0, 1]`.

Which class to use:

- Upscale by 4 or 2 without resizing the input first, with Real-ESRGAN weights: :class:`RRDBNetBuilder`.
- Upscale a 224 x 224 resize of every input by 3 with a small network: :class:`SmallSRBuilder`.
- Pick either builder from one dataclass: :class:`SuperResolutionConfig`, passed to
  ``SuperResolution.from_config``.
- Your own super-resolution network with its own pre- and post-processing: :class:`SuperResolution`.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn

from kornia.core.download import load_state_dict_from_url
from kornia.core.external import PILImage as Image
from kornia.core.external import onnx
from kornia.core.mixin.onnx import ONNXExportMixin, _has_forward_hooks, _preserved_rng, _returns_per_image_outputs
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
    """Choose a super-resolution model by name, to build with SuperResolution.from_config.

    ``sr = SuperResolution.from_config(SuperResolutionConfig("RealESRGAN_x4plus"))`` builds that model with
    :meth:`RRDBNetBuilder.build`; ``"small_sr"``, the default, builds with :meth:`SmallSRBuilder.build`. The fields
    are the arguments of those builders: ``upscale_factor`` and ``image_size`` apply only to ``"small_sr"``.

    Args:
        model_name: Name of the model variant to build. RRDB variants are
            ``"RealESRGAN_x4plus"``, ``"RealESRNet_x4plus"``,
            ``"RealESRGAN_x4plus_anime_6B"`` and ``"RealESRGAN_x2plus"``, spelled exactly;
            the lightweight variant is ``"small_sr"``, in any case.
        pretrained: If ``True``, load pretrained weights for the selected variant.
        upscale_factor: Integer scale of ``"small_sr"`` (see :meth:`SmallSRBuilder.build`). Ignored by
            the RRDB variants, whose scale is fixed by ``model_name``.
        image_size: The fixed input size of ``"small_sr"`` for ONNX export (see :meth:`SmallSRBuilder.build`).

    See Also:
        - :class:`SuperResolution`: the model that :meth:`SuperResolution.from_config` builds from this.
        - :class:`RRDBNetBuilder`: builds the RRDB variants directly.
        - :class:`SmallSRBuilder`: builds ``"small_sr"`` directly.

    """

    model_name: str = "small_sr"
    pretrained: bool = True
    upscale_factor: int = 3
    image_size: Optional[int] = None


class SuperResolution(ModelBase[SuperResolutionConfig], ONNXExportMixin):
    """Run a super-resolution network on RGB images with its pre- and post-processing.

    ``sr = SuperResolution(model, pre_processor, post_processor)`` wraps a super-resolution network.
    ``out = sr(images)`` takes a ``(B, 3, H, W)`` tensor or a list of ``(3, H, W)`` images, runs
    ``pre_processor``, ``model`` and ``post_processor`` in turn, and returns what the post-processor returns. For the
    pretrained models, use :class:`RRDBNetBuilder`, :class:`SmallSRBuilder` or :meth:`from_config`.

    See Also:
        - :class:`RRDBNetBuilder`: Real-ESRGAN weights, x4 or x2, without resizing the input first.
        - :class:`SmallSRBuilder`: a small network that upscales a 224 x 224 resize by 3.
        - :class:`SuperResolutionConfig`: picks either builder by name, for :meth:`from_config`.
        - :func:`~kornia.geometry.transform.resize`: interpolation to any size, without a network.

    """

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
            images: If list of RGB images. Each image is a torch.Tensor with shape ``(3, H, W)``.
                If torch.Tensor, a torch.Tensor with shape ``(B, 3, H, W)``.

        Returns:
            What the post-processor returns: for the builders' models, see :class:`RRDBNetBuilder` and
            :class:`SmallSRBuilder`.

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
        """Collect the super-resolved images for display.

        Args:
            images: input torch.Tensor.
            edge_maps: super-resolved images; computed with :meth:`forward` when None.
            output_type: ``"torch"`` or ``"pil"``.

        Returns:
            A list with one ``(3, H', W')`` image per super-resolved image, or a list of PIL images for
            ``output_type="pil"``.

        """
        if edge_maps is None:
            edge_maps = self.forward(images)
        output = []
        for edge_map in edge_maps:
            output.append(edge_map)

        return self._tensor_to_type(output, output_type)

    def save(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        edge_maps: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        directory: Optional[str] = None,
        output_type: str = "torch",
    ) -> None:
        """Save the input images and their super-resolved images as PNG files.

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

        Returns:
            The exported ONNX model.

        Raises:
            ValueError: With the pre- and post-processor, if the output is a list (or tuple) with one image per input
                image and the exported batch is dynamic, from a ``dynamic_axes`` or ``dynamic_shapes`` argument or,
                with the TorchScript exporter, because the list comes only for more than one image.

        Note:
            With the pre- and post-processor, an output that is a list with one image per input image, or a tuple as
            long as the batch at two batch sizes, is exported for a batch of one image: the exporters turn such a
            sequence into one graph output per traced image, so a dynamic batch would return the traced batch's
            images at any batch size. The list comes from a post-processor that
            returns one, from the network itself, or from a pre-processor that returns a list of images, which
            :meth:`forward` runs through the network one at a time. Known tensor-output networks (the builders'
            RRDB and SmallSR networks, ``nn.Identity`` and ``nn.Conv2d``) with ``nn.Identity`` or ``ResizePreProcessor``
            before them and ``nn.Identity`` or ``OutputRangePostProcessor`` after them keep a dynamic batch without a
            probe, provided the wrapper and each component retain their known ``forward`` implementations and no
            forward hook applies to them, a global one included. Other combinations are probed on one image at the
            export size in eval mode, without gradients and outside the export. A list result requires a fixed batch;
            a tensor result is checked again on the actual tracing batch before accepting a dynamic graph. A dynamic
            graph with per-image outputs raises a ``ValueError`` after the export, before anything is saved.

        """
        if onnx_name is None:
            onnx_name = f"kornia_{self.name}.onnx"

        # The dummy input must agree with a fixed input size, which ``to_onnx`` checks.
        pseudo_size = self.pseudo_image_size or self.input_image_size or 352
        pseudo_shape = [1, 3, pseudo_size, pseudo_size]
        # Exported for a dynamic batch, a list of per-image outputs comes out as the traced batch's images, or the
        # export fails: such a model is exported for one image.
        per_image = None
        if include_pre_and_post_processor and not self._returns_one_tensor_without_a_probe():
            # The probe's input draws from the global generator, as the export's dummy does, and gives back what it
            # drew.
            with _preserved_rng():
                per_image = _returns_per_image_outputs(self, torch.rand(pseudo_shape))
        batch = 1 if per_image else -1

        with self._onnx_output_kind_kept(per_image):
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
        # A plain callable as a processor has no ``forward``: it is probed.
        forward = getattr(self.forward, "__func__", None)
        network = getattr(getattr(self.model, "forward", None), "__func__", None)
        pre = getattr(getattr(self.pre_processor, "forward", None), "__func__", None)
        post = getattr(getattr(self.post_processor, "forward", None), "__func__", None)
        return (
            not _has_forward_hooks(self)
            and forward is SuperResolution.forward
            and network in (nn.Identity.forward, nn.Conv2d.forward, RRDBNet.forward, SmallSRNetWrapper.forward)
            and pre in (nn.Identity.forward, ResizePreProcessor.forward)
            and post in (nn.Identity.forward, OutputRangePostProcessor.forward)
        )

    def _onnx_fixed_batch_hint(self, module: Any, dummy_input: Any) -> Optional[str]:
        if (
            module is not self
            or self._returns_one_tensor_without_a_probe()
            or not self._onnx_returns_per_image_outputs(dummy_input)
        ):
            return None
        return (
            "Use a network that returns one tensor for the batch, with a pre-processor that hands it one tensor and "
            "a post-processor that keeps it one, such as nn.Identity or ResizePreProcessor and "
            "OutputRangePostProcessor, to export a dynamic batch."
        )


class RRDBNetBuilder:
    """Build a SuperResolution model around a Real-ESRGAN RRDB generator that upscales by 4 or 2.

    ``sr = RRDBNetBuilder.build("RealESRNet_x4plus")`` downloads the released Real-ESRGAN weights on first use and
    returns a :class:`SuperResolution`. ``out = sr(images)`` upscales RGB images without resizing them first: a
    ``(B, 3, H, W)`` tensor becomes ``(B, 3, 4 H, 4 W)``, or ``(B, 3, 2 H, 2 W)`` for ``"RealESRGAN_x2plus"``,
    clamped to :math:`[0, 1]`. A list of ``(3, H, W)`` images, of any sizes, returns a list.

    See Also:
        - :class:`SmallSRBuilder`: a small network that upscales a 224 x 224 resize by 3.
        - :class:`~kornia.models.RRDBNet`: the generator itself, to train or to load other weights.
        - :class:`SuperResolutionConfig`: picks this builder or the other by name.

    """

    @staticmethod
    def build(model_name: str = "RealESRNet_x4plus", pretrained: bool = True) -> SuperResolution:
        """Build a Real-ESRGAN super-resolution model.

        Args:
            model_name: Name of the RRDB variant to construct, spelled exactly: ``"RealESRGAN_x4plus"`` and
                ``"RealESRNet_x4plus"`` (23 blocks, x4), ``"RealESRGAN_x4plus_anime_6B"`` (6 blocks, x4) and
                ``"RealESRGAN_x2plus"`` (23 blocks, x2), which needs even ``H`` and ``W``.
            pretrained: If ``True``, download and load pretrained weights for the
                selected model variant; otherwise the network is randomly initialized.

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
    """Build a SuperResolution model around a small sub-pixel network that upscales a 224 x 224 resize.

    ``sr = SmallSRBuilder.build()`` downloads the weights on first use and returns a :class:`SuperResolution`.
    ``out = sr(images)`` resizes each RGB image in :math:`[0, 1]` to 224 x 224, discarding its aspect ratio, and
    returns one ``(B, 3, 224 r, 224 r)`` tensor in :math:`[0, 1]`, ``r = upscale_factor``.

    See Also:
        - :class:`RRDBNetBuilder`: Real-ESRGAN weights, x4 or x2, keeping the aspect ratio.
        - :class:`SuperResolutionConfig`: picks this builder or the other by name.

    """

    @staticmethod
    def build(
        model_name: str = "small_sr", pretrained: bool = True, upscale_factor: int = 3, image_size: Optional[int] = None
    ) -> SuperResolution:
        """Build the small super-resolution model.

        Args:
            model_name: Name of the model variant to build. Only ``"small_sr"`` (any case) is supported.
            pretrained: If ``True``, load pretrained weights for the selected model.
            upscale_factor: Integer scale used by ``SmallSRNetWrapper``. The pre-processor
                first resizes every input to 224 x 224, discarding its aspect ratio, so the
                output is ``224 * upscale_factor`` square: with ``3``, a 100 x 50 and a
                480 x 640 input both give 672 x 672. Pretrained weights exist only for
                ``3``; another factor with ``pretrained=True`` raises a size-mismatch
                :class:`RuntimeError`.
            image_size: Optional fixed input height and width of the graph that
                :meth:`SuperResolution.to_onnx` exports; ``None`` leaves them dynamic. It does not
                change the pre-processor, which always resizes to 224, so the output side stays
                ``224 * upscale_factor``.

        Returns:
            A :class:`SuperResolution` instance configured with resizing pre-processing
            and identity post-processing, the network's output being in :math:`[0, 1]` already.

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
