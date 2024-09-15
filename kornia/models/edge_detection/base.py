from typing import Union

from kornia.core import Tensor
from kornia.core.external import PILImage as Image
from kornia.models.base import ModelBase
__all__ = ["EdgeDetector"]


class EdgeDetector(ModelBase):
    """EdgeDetector is a module that wraps an edge detection model.

    This module uses EdgeDetectionModel library for edge detection.
    """

    def forward(self, images: Union[Tensor, list[Tensor]]) -> Tensor:
        """Forward pass of the semantic segmentation model.

        Args:
            images: input tensor.

        Returns:
            output tensor.
        """
        images, image_sizes = self.pre_processor(images)
        out_images = self.model(images)
        return self.post_processor(out_images, image_sizes)

    def draw(
        self, images: Union[Tensor, list[Tensor]], output_type: str = "torch"
    ) -> Union[Tensor, list[Tensor], list[Image.Image]]:  # type: ignore
        """Draw the segmentation results.

        Args:
            images: input tensor.
            output_type: type of the output.

        Returns:
            output tensor.
        """
        ...

    def save(self, images: Union[Tensor, list[Tensor]], output_type: str = "torch") -> None:
        """Save the segmentation results.

        Args:
            images: input tensor.
            output_type: type of the output.

        Returns:
            output tensor.
        """
        ...
