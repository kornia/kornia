from typing import Union

import torch

from kornia.core import Tensor
from kornia.core.external import PILImage as Image
from kornia.models.base import ModelBase


class SemanticSegmentation(ModelBase):
    """Semantic Segmentation is a module that wraps a semantic segmentation model.

    This module uses SegmentationModel library for semantic segmentation.
    """

    @torch.inference_mode()
    def forward(self, images: Union[Tensor, list[Tensor]]) -> Union[Tensor, list[Tensor]]:
        """Forward pass of the semantic segmentation model.

        Args:
            images: If list of RGB images. Each image is a Tensor with shape :math:`(3, H, W)`.
                If Tensor, a Tensor with shape :math:`(B, 3, H, W)`.

        Returns:
            output tensor.
        """
        images = self.pre_processor(images)
        output = self.model(images)
        return self.post_processor(output)

    def visualize(
        self, images: Union[Tensor, list[Tensor]], output_type: str = "torch"
    ) -> Union[Tensor, list[Tensor], list[Image.Image]]:  # type: ignore
        """Draw the segmentation results.

        Args:
            images: input tensor.
            output_type: type of the output.

        Returns:
            output tensor.
        """
        raise NotImplementedError("Visualization is not implemented for this model.")

    def save(self, images: Union[Tensor, list[Tensor]], output_type: str = "torch") -> None:
        """Save the segmentation results.

        Args:
            images: input tensor.
            output_type: type of the output.

        Returns:
            output tensor.
        """
        raise NotImplementedError("Saving is not implemented for this model.")
