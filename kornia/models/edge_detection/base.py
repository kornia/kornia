from typing import Optional, Union

from kornia.color.gray import grayscale_to_rgb
from kornia.core import Tensor
from kornia.core.external import PILImage as Image
from kornia.models.base import ModelBase

__all__ = ["EdgeDetector"]


class EdgeDetector(ModelBase):
    """EdgeDetector is a module that wraps an edge detection model.

    This module uses EdgeDetectionModel library for edge detection.
    """

    name: str = "edge_detection"

    def forward(self, images: Union[Tensor, list[Tensor]]) -> Union[Tensor, list[Tensor]]:
        """Forward pass of the semantic segmentation model.

        Args:
            images: input tensor.

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
        """Draw the segmentation results.

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
        """Save the segmentation results.

        Args:
            images: input tensor.
            output_type: type of the output.

        Returns:
            output tensor.
        """
        outputs = self.visualize(images, edge_maps, output_type)
        self._save_outputs(images, directory, suffix="_src")
        self._save_outputs(outputs, directory, suffix="_edge")
