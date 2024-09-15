from typing import Union

from kornia.core import Module, Tensor
from kornia.core.external import PILImage as Image

__all__ = ["EdgeDetector"]


class EdgeDetector(Module):
    """EdgeDetector is a module that wraps an edge detection model.

    This module uses EdgeDetectionModel library for edge detection.
    """

    # NOTE: We need to implement this class for better visualization and user experience.

    def __init__(self, model: Module, pre_processor: Module, post_processor: Module) -> None:
        """Construct an Object Detector object.

        Args:
            model: an object detection model.
            pre_processor: a pre-processing module
        """
        super().__init__()
        self.model = model
        self.pre_processor = pre_processor
        self.post_processor = post_processor

    def forward(self, images: Union[Tensor, list[Tensor]]) -> Tensor:
        """Forward pass of the semantic segmentation model.

        Args:
            x: input tensor.

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
