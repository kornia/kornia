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

from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch import nn

from kornia.color.gray import grayscale_to_rgb
from kornia.color.rgb import RgbToBgr
from kornia.core.external import PILImage as Image
from kornia.core.external import onnx
from kornia.core.mixin.onnx import ONNXExportMixin
from kornia.enhance.normalize import Normalize
from kornia.geometry.transform import resize
from kornia.models.base import ModelBase
from kornia.models.dexined import DexiNed
from kornia.models.processors import ResizePostProcessor, ResizePreProcessor

__all__ = ["EdgeDetector", "EdgeDetectorBuilder"]


class EdgeDetector(ModelBase, ONNXExportMixin):
    """EdgeDetector is a module that wraps an edge detection model.

    This is a high-level API that wraps edge detection models like :py:class:`kornia.models.DexiNed`.
    :py:class:`~kornia.contrib.edge_detection.EdgeDetectorBuilder` builds one with the input conversion, the network
    and the sigmoid in ``model``.

    Args:
        model: The edge detection model.
        pre_processor: Pre-processing module (e.g., ResizePreProcessor).
        post_processor: Post-processing module (e.g., ResizePostProcessor).
        name: Optional name for the detector.

    Example:
        >>> detector = EdgeDetectorBuilder.build("dexined", pretrained=True, image_size=352)
        >>> img = torch.rand(1, 3, 320, 320)  # RGB in [0, 1]
        >>> out = detector(img)  # a list with one (1, 1, 320, 320) edge probability map per image

    """

    name: str = "edge_detection"
    # Set by ``to_onnx`` for the duration of its export, read by ``forward`` (see there).
    _exporting_to_onnx: bool = False

    def __init__(
        self,
        model: torch.nn.Module,
        pre_processor: torch.nn.Module,
        post_processor: torch.nn.Module,
        name: Optional[str] = None,
    ) -> None:
        """Initialize EdgeDetector.

        Args:
            model: The edge detection model.
            pre_processor: Pre-processing module (e.g., ResizePreProcessor).
            post_processor: Post-processing module (e.g., ResizePostProcessor).
            name: Optional name for the detector.

        """
        super().__init__()
        self.model = model.eval()
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        if name is not None:
            self.name = name

    @staticmethod
    def from_config(config: Any) -> EdgeDetector:
        """Build EdgeDetector from config.

        This is a placeholder to satisfy the abstract method requirement.
        Use EdgeDetectorBuilder.build() or instantiate EdgeDetector directly.

        Args:
            config: Configuration object (not used, kept for interface compatibility).

        Returns:
            EdgeDetector instance.

        """
        raise NotImplementedError(
            "EdgeDetector.from_config() is not implemented. "
            "Use EdgeDetectorBuilder.build() or instantiate EdgeDetector directly."
        )

    @torch.inference_mode()
    def forward(self, images: Union[torch.Tensor, list[torch.Tensor]]) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Forward pass of the edge detection model.

        Args:
            images: If list of RGB images. Each image is a torch.Tensor with shape :math:`(3, H, W)`.
                If torch.Tensor, a torch.Tensor with shape :math:`(B, 3, H, W)`. For a detector from
                :py:class:`~kornia.contrib.edge_detection.EdgeDetectorBuilder`, the pixel values are in
                :math:`[0, 1]`.

        Returns:
            A list with one output of shape :math:`(1, 1, H, W)` per input image, resized back to that image's size.
            For a detector from :py:class:`~kornia.contrib.edge_detection.EdgeDetectorBuilder`, it holds edge
            probabilities in :math:`[0, 1]`. Under an ONNX export with a ``ResizePostProcessor``, one
            :math:`(B, 1, H, W)` tensor at the input's size instead, see :meth:`to_onnx`.

        Raises:
            ValueError: Under an ONNX export through :meth:`to_onnx`, or a direct one on the TorchScript exporter or,
                from torch 2.9, the dynamo exporter, if the ``ResizePostProcessor`` resizes with ``"area"``.

        """
        if (
            (self._exporting_to_onnx or torch.onnx.is_in_onnx_export())
            and isinstance(images, torch.Tensor)
            and self._resizes_back_on_export()
        ):
            # An exported graph returns one tensor, so the maps of a batch, whose images share a size, are resized
            # back together: the ``torch.cat`` of the eager list. The size is read from the input, a symbolic size
            # under every exporter, not from the pre-processor's ``original_sizes``, whose values
            # ``ResizePostProcessor`` reads with ``.tolist()``. The flag covers the dynamo exporter below torch 2.9,
            # where ``is_in_onnx_export()`` is False; ``is_exporting()`` would also be True under ``torch.compile``
            # there (#5037).
            interpolation = self._export_interpolation()
            height, width = images.shape[-2:]
            out_images = self.model(self.pre_processor(images)[0])
            return resize(out_images, (height, width), interpolation=interpolation)
        images, image_sizes = self.pre_processor(images)
        out_images = self.model(images)
        return self.post_processor(out_images, image_sizes)

    def _resizes_back_on_export(self) -> bool:
        """Whether an export replaces the post-processor: a ``ResizePostProcessor`` that keeps its own ``forward``.

        A subclass that overrides ``forward`` is exported as that ``forward`` computes.
        """
        return (
            isinstance(self.post_processor, ResizePostProcessor)
            and type(self.post_processor).forward is ResizePostProcessor.forward
        )

    def _export_interpolation(self) -> str:
        """The post-processor's interpolation mode, which an export resizes the maps back with.

        Raises:
            ValueError: For ``"area"``, which an exported graph cannot compute at the input's size.
        """
        interpolation = self.post_processor.interpolation_mode
        if interpolation == "area":
            raise ValueError(
                'A ResizePostProcessor with interpolation_mode="area" cannot be exported with the input\'s size: '
                '"area" resizing is adaptive average pooling, which an exported graph cannot compute to a size known '
                'only at run time. Build the post-processor with "bilinear", "bicubic", "nearest" or "nearest-exact" '
                "for export."
            )
        return interpolation

    def visualize(
        self,
        images: Union[torch.Tensor, list[torch.Tensor]],
        edge_maps: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        output_type: str = "torch",
    ) -> Union[torch.Tensor, list[torch.Tensor], list[Image.Image]]:
        """Draw the edge detection results.

        Args:
            images: input torch.Tensor.
            edge_maps: detected edges.
            output_type: type of the output.

        Returns:
            output torch.Tensor.

        """
        if edge_maps is None:
            edge_maps = self.forward(images)
        output = []
        for edge_map in edge_maps:
            output.append(grayscale_to_rgb(edge_map)[0])

        return self._tensor_to_type(output, output_type, is_batch=isinstance(images, torch.Tensor))

    def save(
        self,
        images: Union[torch.Tensor, list[torch.Tensor]],
        edge_maps: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        directory: Optional[str] = None,
        output_type: str = "torch",
    ) -> None:
        """Save the edge detection results.

        Args:
            images: input torch.Tensor.
            edge_maps: detected edges.
            output_type: type of the output.
            directory: where to save outputs.

        Returns:
            output torch.Tensor.

        """
        outputs = self.visualize(images, edge_maps, output_type)
        self._save_outputs(images, directory, suffix="_src")
        self._save_outputs(outputs, directory, suffix="_edge")

    def to_onnx(
        self,
        onnx_name: Optional[str] = None,
        image_size: Optional[int] = 352,
        include_pre_and_post_processor: bool = True,
        save: bool = True,
        additional_metadata: Optional[list[tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> onnx.ModelProto:
        """Export the current edge detection model to an ONNX model file.

        The graph takes one :math:`(B, 3, H, W)` tensor. With the pre- and post-processor and a
        ``ResizePostProcessor``, it returns the edge maps of the batch as one :math:`(B, 1, H, W)` tensor at the
        input's size, resized with the post-processor's ``"bilinear"``, ``"bicubic"``, ``"nearest"`` or
        ``"nearest-exact"`` mode: the ``torch.cat`` of the list that :meth:`forward` returns. A subclass of
        ``ResizePostProcessor`` that overrides ``forward`` is exported as its ``forward`` computes. Export through
        this method rather than ``torch.onnx.export(detector, ...)``: below torch 2.9, a direct
        ``torch.onnx.export(..., dynamo=True)`` traces the eager per-image list instead.

        Args:
            onnx_name:
                The name of the output ONNX file. If not provided, ``"kornia_<name>_<image_size>.onnx"``, for
                example ``"kornia_dexined_352.onnx"``.
            image_size:
                The height and width of the graph's input, which the returned maps have too. If None, both are
                dynamic. The pre-processor keeps the size it was built with, so the model runs at that size
                whatever ``image_size`` is.
            include_pre_and_post_processor:
                Whether to include the pre-processor and post-processor in the exported model.
            save:
                If True, write the model to ``onnx_name``; it is returned either way.
            additional_metadata:
                Additional metadata to add to the ONNX model.
            kwargs: Additional keyword arguments of :func:`torch.onnx.export`.

        Raises:
            ValueError: With the pre- and post-processor, if the ``ResizePostProcessor`` resizes with ``"area"``:
                adaptive average pooling cannot be exported with the input's size.

        """
        if onnx_name is None:
            onnx_name = f"kornia_{self.name}_{image_size}.onnx"
        if include_pre_and_post_processor and self._resizes_back_on_export():
            self._export_interpolation()  # raises before the export for a mode it cannot export

        previous = self._exporting_to_onnx
        self._exporting_to_onnx = True
        try:
            return ONNXExportMixin.to_onnx(
                self,
                onnx_name,
                input_shape=[-1, 3, image_size or -1, image_size or -1],
                output_shape=[-1, 1, image_size or -1, image_size or -1],
                pseudo_shape=[1, 3, image_size or 352, image_size or 352],
                model=self if include_pre_and_post_processor else self.model,
                save=save,
                additional_metadata=additional_metadata,
                **kwargs,
            )
        finally:
            self._exporting_to_onnx = previous


class EdgeDetectorBuilder:
    """EdgeDetectorBuilder is a class that builds an edge detection model.

    This is a high-level API that builds edge detection models like :py:class:`kornia.models.DexiNed`
    and wraps them with :py:class:`EdgeDetector`.

    Note:
        To use this model, load image tensors and call ``model.save(images)``.
    """

    @staticmethod
    def build(model_name: str = "dexined", pretrained: bool = True, image_size: int = 352) -> EdgeDetector:
        """Build an edge detection model.

        The detector takes RGB images in :math:`[0, 1]`. For ``"dexined"`` it flips them to BGR, scales them to
        :math:`[0, 255]` and subtracts the BGR mean ``[103.939, 116.779, 123.68]``, the input the pretrained DexiNed
        was trained on (xavysp/DexiNed@d944b70), then applies a sigmoid to the network's logits.

        Args:
            model_name: Name of the model to build. Currently only "dexined" is supported.
            pretrained: If True, loads pretrained weights.
            image_size: Size to which input images will be resized during preprocessing.

        Returns:
            EdgeDetector instance configured with the specified model.

        Example:
            >>> detector = EdgeDetectorBuilder.build(pretrained=True, image_size=352)
            >>> img = torch.rand(1, 3, 320, 320)
            >>> out = detector(img)

        """
        if model_name.lower() == "dexined":
            # RGB in [0, 1] -> 255 * BGR - mean_bgr, with mean_bgr in B, G, R order (xavysp/DexiNed@d944b70
            # main.py:319-320). Both steps share slot 0, so DexiNed stays at ``model[1]`` and the state-dict keys stay
            # ``model.1.*``.
            mean_bgr = torch.tensor([[103.939, 116.779, 123.68]])
            to_bgr = nn.Sequential(RgbToBgr(), Normalize(mean=mean_bgr / 255.0, std=torch.tensor([[1.0 / 255.0] * 3])))
            model = nn.Sequential(to_bgr, DexiNed(pretrained=pretrained), nn.Sigmoid())
        else:
            raise ValueError(f"Model {model_name} not found. Please choose from 'dexined'.")

        return EdgeDetector(
            model,
            ResizePreProcessor(image_size, image_size),
            ResizePostProcessor(),
            name="dexined",
        )
