from typing import Union

from kornia.core import Tensor
from kornia.core.external import PILImage as Image
from kornia.models.base import ModelBase


class SemanticSegmentation(ModelBase):
    """Semantic Segmentation is a module that wraps a semantic segmentation model.

    This module uses SegmentationModel library for semantic segmentation.
    """

    def forward(self, images: Union[Tensor, list[Tensor]]) -> Tensor:
        """Forward pass of the semantic segmentation model.

        Args:
            x: input tensor.

        Returns:
            output tensor.
        """
        images = self.pre_processor(images)
        output = self.model(images)
        return self.post_processor(output)

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
