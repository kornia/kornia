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
from kornia.core.external import PILImage as Image
from kornia.core.external import onnx
from kornia.core.mixin.onnx import ONNXExportMixin
from kornia.enhance.normalize import Normalize
from kornia.models.base import ModelBase
from kornia.models.dexined import DexiNed
from kornia.models.processors import ResizePostProcessor, ResizePreProcessor

__all__ = ["EdgeDetector", "EdgeDetectorBuilder"]


class EdgeDetector(ModelBase, ONNXExportMixin):
    """EdgeDetector is a module that wraps an edge detection model.

    This is a high-level API that wraps edge detection models like :py:class:`kornia.models.DexiNed`.

    Args:
        model: The edge detection model.
        pre_processor: Pre-processing module (e.g., ResizePreProcessor).
        post_processor: Post-processing module (e.g., ResizePostProcessor).
        name: Optional name for the detector.

    Example:
        >>> from kornia.models.dexined import DexiNed
        >>> from kornia.models.processors import ResizePreProcessor, ResizePostProcessor
        >>> model = DexiNed(pretrained=True)
        >>> detector = EdgeDetector(model, ResizePreProcessor(352, 352), ResizePostProcessor())
        >>> img = torch.rand(1, 3, 320, 320)
        >>> out = detector(img)

    """

    name: str = "edge_detection"

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
                If torch.Tensor, a torch.Tensor with shape :math:`(B, 3, H, W)`.

        Returns:
            output torch.Tensor.

        """
        images, image_sizes = self.pre_processor(images)
        out_images = self.model(images)
        return self.post_processor(out_images, image_sizes)

    def visualize(
        self,
        images: Union[torch.Tensor, list[torch.Tensor]],
        edge_maps: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        output_type: str = "torch",
    ) -> Union[torch.Tensor, list[torch.Tensor], list[Image.Image]]:  # type: ignore
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

    def to_onnx(  # type: ignore[override]
        self,
        onnx_name: Optional[str] = None,
        image_size: Optional[int] = 352,
        include_pre_and_post_processor: bool = True,
        save: bool = True,
        additional_metadata: Optional[list[tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> onnx.ModelProto:  # type: ignore
        """Export the current edge detection model to an ONNX model file.

        Args:
            onnx_name:
                The name of the output ONNX file. If not provided, a default name in the
                format "Kornia-<ClassName>.onnx" will be used.
            image_size:
                The size to which input images will be resized during preprocessing.
                If None, image_size will be dynamic. For DexiNed, recommended scale is 352.
            include_pre_and_post_processor:
                Whether to include the pre-processor and post-processor in the exported model.
            save:
                If to save the model or load it.
            additional_metadata:
                Additional metadata to add to the ONNX model.
            kwargs: Additional arguments to convert to onnx.

        """
        if onnx_name is None:
            onnx_name = f"kornia_{self.name}_{image_size}.onnx"

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
            # Normalize then scale to [0, 255]
            norm = Normalize(mean=torch.tensor([[0.485, 0.456, 0.406]]), std=torch.tensor([[1.0 / 255.0] * 3]))
            model = nn.Sequential(norm, DexiNed(pretrained=pretrained), nn.Sigmoid())
        else:
            raise ValueError(f"Model {model_name} not found. Please choose from 'dexined'.")

        return EdgeDetector(
            model,
            ResizePreProcessor(image_size, image_size),
            ResizePostProcessor(),
            name="dexined",
        )
