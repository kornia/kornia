from typing import ClassVar, List, Optional, Union

import torch

import kornia
from kornia.core import Tensor
from kornia.core.external import PILImage as Image
from kornia.models.base import ModelBase


class SemanticSegmentation(ModelBase):
    """Semantic Segmentation is a module that wraps a semantic segmentation model.

    This module uses SegmentationModel library for semantic segmentation.
    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[List[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[List[int]] = [-1, -1, -1, -1]

    @torch.inference_mode()
    def forward(self, images: Union[Tensor, list[Tensor]]) -> Union[Tensor, list[Tensor]]:
        """Forward pass of the semantic segmentation model.

        Args:
            images: If list of RGB images. Each image is a Tensor with shape :math:`(3, H, W)`.
                If Tensor, a Tensor with shape :math:`(B, 3, H, W)`.

        Returns:
            output tensor.
        """
        if isinstance(
            images,
            (
                list,
                tuple,
            ),
        ):
            images = torch.stack(images, dim=0)

        images = self.pre_processor(images)
        output = self.model(images)
        return self.post_processor(output)

    def get_colormap(self, num_classes: int, colormap: str = "random", manual_seed: int = 2147) -> Tensor:
        """Get a color map of size num_classes.

        Args:
            num_classes: The number of colors in the color map.
            colormap: The colormap to use, can be "random" or a custom color map.
            manual_seed: The manual seed to use for the colormap.

        Returns:
            A tensor of shape (num_classes, 3) representing the color map.
        """
        if colormap == "random":
            # Generate a color for each class
            g_cpu = torch.Generator()
            g_cpu.manual_seed(manual_seed)
            colors = torch.rand(num_classes, 3, generator=g_cpu)
        else:
            raise ValueError(f"Unsupported colormap: {colormap}")

        return colors

    def visualize(
        self,
        images: Union[Tensor, list[Tensor]],
        semantic_masks: Optional[Union[Tensor, list[Tensor]]] = None,
        output_type: str = "torch",
        colormap: str = "random",
        manual_seed: int = 2147,
    ) -> Union[Tensor, list[Tensor], list["Image.Image"]]:  # type: ignore
        """Visualize the segmentation masks.

        Args:
            images: If list of RGB images. Each image is a Tensor with shape :math:`(3, H, W)`.
                If Tensor, a Tensor with shape :math:`(B, 3, H, W)`.
            semantic_masks: If list of segmentation masks. Each mask is a Tensor with shape :math:`(C, H, W)`.
                If Tensor, a Tensor with shape :math:`(B, C, H, W)`.
            output_type: The type of output, can be "torch" or "PIL".
            colormap: The colormap to use, can be "random" or a custom color map.
            manual_seed: The manual seed to use for the colormap.
        """
        if semantic_masks is None:
            semantic_masks = self.forward(images)

        if isinstance(
            semantic_masks,
            (
                list,
                tuple,
            ),
        ):
            semantic_masks = torch.stack(semantic_masks, dim=0)

        # Generate a color for each class
        colors = self.get_colormap(semantic_masks.size(1), colormap, manual_seed=manual_seed)

        if torch.allclose(
            semantic_masks.sum(dim=1), torch.tensor(1, dtype=semantic_masks.dtype, device=semantic_masks.device)
        ):
            # Softmax is used, thus, muliclass segmentation
            semantic_masks = semantic_masks.argmax(dim=1, keepdim=True)
            # Create a colormap for each pixel based on the class with the highest probability
            output = colors[semantic_masks.squeeze(1)]
            output = output.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "Only muliclass segmentation is supported. Please ensure a softmax is used, or submit a PR."
            )

        return self._tensor_to_type(output, output_type, is_batch=True)

    def save(
        self,
        images: Union[Tensor, list[Tensor]],
        semantic_masks: Optional[Union[Tensor, list[Tensor]]] = None,
        directory: Optional[str] = None,
        output_type: str = "torch",
        colormap: str = "random",
        manual_seed: int = 2147,
    ) -> None:
        """Save the segmentation results.

        Args:
            images: If list of RGB images. Each image is a Tensor with shape :math:`(3, H, W)`.
                If Tensor, a Tensor with shape :math:`(B, 3, H, W)`.
            semantic_masks: If list of segmentation masks. Each mask is a Tensor with shape :math:`(C, H, W)`.
                If Tensor, a Tensor with shape :math:`(B, C, H, W)`.
            directory: The directory to save the results.
            output_type: The type of output, can be "torch" or "PIL".
            colormap: The colormap to use, can be "random" or a custom color map.
            manual_seed: The manual seed to use for the colormap.
        """

        colored_masks = self.visualize(images, semantic_masks, output_type, colormap=colormap, manual_seed=manual_seed)
        if isinstance(images, Tensor):
            overlayed = kornia.enhance.add_weighted(images, 0.5, colored_masks, 0.5, 1.0)
        elif isinstance(images, (list, tuple,)):
            overlayed = []
            for i in range(len(images)):
                overlayed.append(kornia.enhance.add_weighted(images[i:i + 1], 0.5, colored_masks[i:i + 1], 0.5, 1.0)[0])
        else:
            raise ValueError(f"`images` should be a Tensor or a list of Tensors. Got {type(images)}")

        self._save_outputs(images, directory, suffix="_src")
        self._save_outputs(colored_masks, directory, suffix="_mask")
        self._save_outputs(overlayed, directory, suffix="_overlay")
