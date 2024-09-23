from __future__ import annotations

from typing import Any, Optional, Union

import torch

from kornia.color.gray import grayscale_to_rgb
from kornia.core import Tensor
from kornia.core.external import PILImage as Image
from kornia.core.external import onnx
from kornia.models.base import ModelBase

__all__ = ["EdgeDetector"]


class EdgeDetector(ModelBase):
    """EdgeDetector is a module that wraps an edge detection model."""

    name: str = "edge_detection"

    @torch.inference_mode()
    def forward(self, images: Union[Tensor, list[Tensor]]) -> Union[Tensor, list[Tensor]]:
        """Forward pass of the edge detection model.

        Args:
            images: If list of RGB images. Each image is a Tensor with shape :math:`(3, H, W)`.
                If Tensor, a Tensor with shape :math:`(B, 3, H, W)`.

        Returns:
            output tensor.
        """
        images, image_sizes = self.pre_processor(images)
        out_images = self.model(images)
        return self.post_processor(out_images, image_sizes)

    def visualize(
        self,
        images: Union[Tensor, list[Tensor]],
        edge_maps: Optional[Union[Tensor, list[Tensor]]] = None,
        output_type: str = "torch",
    ) -> Union[Tensor, list[Tensor], list[Image.Image]]:  # type: ignore
        """Draw the edge detection results.

        Args:
            images: input tensor.
            output_type: type of the output.

        Returns:
            output tensor.
        """
        if edge_maps is None:
            edge_maps = self.forward(images)
        output = []
        for edge_map in edge_maps:
            output.append(grayscale_to_rgb(edge_map)[0])

        return self._tensor_to_type(output, output_type, is_batch=isinstance(images, Tensor))

    def save(
        self,
        images: Union[Tensor, list[Tensor]],
        edge_maps: Optional[Union[Tensor, list[Tensor]]] = None,
        directory: Optional[str] = None,
        output_type: str = "torch",
    ) -> None:
        """Save the edge detection results.

        Args:
            images: input tensor.
            output_type: type of the output.

        Returns:
            output tensor.
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
        additional_metadata: list[tuple[str, str]] = [],
        **kwargs: Any,
    ) -> onnx.ModelProto:  # type: ignore
        """Exports the current edge detection model to an ONNX model file.

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
        """
        if onnx_name is None:
            onnx_name = f"kornia_{self.name}_{image_size}.onnx"

        return super().to_onnx(
            onnx_name,
            input_shape=[-1, 3, image_size or -1, image_size or -1],
            output_shape=[-1, 1, image_size or -1, image_size or -1],
            pseudo_shape=[1, 3, image_size or 352, image_size or 352],
            model=self if include_pre_and_post_processor else self.model,
            save=save,
            additional_metadata=additional_metadata,
            **kwargs,
        )
