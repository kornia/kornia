from typing import Union

from kornia.core import Module, Tensor
from kornia.core.external import PILImage as Image


class SemanticSegmentation(Module):
    """Semantic Segmentation is a module that wraps a semantic segmentation model.

    This module uses SegmentationModel library for semantic segmentation.
    """
    # NOTE: We need to implement this class for better visualization and user experience.

    def __init__(self, model: Module, pre_processor: Module) -> None:
        """Construct an Object Detector object.

        Args:
            model: an object detection model.
            pre_processor: a pre-processing module
        """
        super().__init__()
        self.model = model
        self.pre_processor = pre_processor

    def forward(self, images: Union[Tensor, list[Tensor]]) -> Tensor:
        """Forward pass of the semantic segmentation model.

        Args:
            x: input tensor.

        Returns:
            output tensor.
        """
        images = self.pre_processor(images)
        return self.model(images)

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

    def save(
        self, images: Union[Tensor, list[Tensor]], output_type: str = "torch"
    ) -> None:
        """Save the segmentation results.

        Args:
            images: input tensor.
            output_type: type of the output.

        Returns:
            output tensor.
        """
        ...
