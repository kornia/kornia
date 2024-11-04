from __future__ import annotations

from typing import ClassVar, Optional, Union

import torch

import kornia
from kornia.core import Tensor
from kornia.core.external import PILImage as Image
from kornia.models.base import ModelBase

__all__ = ["SemanticSegmentation"]


class SemanticSegmentation(ModelBase):
    """Semantic Segmentation is a module that wraps a semantic segmentation model.

    This module uses SegmentationModel library for semantic segmentation.
    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, -1, -1, -1]

    @torch.inference_mode()
    def forward(self, images: Union[Tensor, list[Tensor]]) -> Union[Tensor, list[Tensor]]:
        """Forward pass of the semantic segmentation model.

        Args:
            images: If list of RGB images. Each image is a Tensor with shape :math:`(3, H, W)`.
                If Tensor, a Tensor with shape :math:`(B, 3, H, W)`.

        Returns:
            output tensor.
        """
        outputs: Union[Tensor, list[Tensor]]

        if isinstance(
            images,
            (
                list,
                tuple,
            ),
        ):
            outputs = []
            for image in images:
                image = self.pre_processor(image[None])
                output = self.model(image)
                output = self.post_processor(output)
                outputs.append(output[0])
        else:
            images = self.pre_processor(images)
            outputs = self.model(images)
            outputs = self.post_processor(outputs)

        return outputs

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

    def visualize_output(self, semantic_mask: Tensor, colors: Tensor) -> Tensor:
        """Visualize the output of the segmentation model.

        Args:
            semantic_mask: The output of the segmentation model. Shape should be (C, H, W) or (B, C, H, W).
            colors: The color map to use for visualizing the output of the segmentation model.
                Shape should be (num_classes, 3).

        Returns:
            A tensor of shape (3, H, W) or (B, 3, H, W) representing the visualized output of the segmentation model.

        Raises:
            ValueError: If the shape of the semantic mask is not of shape (C, H, W) or (B, C, H, W).
            ValueError: If the shape of the colors is not of shape (num_classes, 3).
            ValueError: If only muliclass segmentation is supported. Please ensure a softmax is used, or submit a PR.
        """
        if semantic_mask.dim() == 3:
            channel_dim = 0
        elif semantic_mask.dim() == 4:
            channel_dim = 1
        else:
            raise ValueError(f"Semantic mask must be of shape (C, H, W) or (B, C, H, W), got {semantic_mask.shape}.")

        if torch.allclose(
            semantic_mask.sum(dim=channel_dim), torch.tensor(1, dtype=semantic_mask.dtype, device=semantic_mask.device)
        ):
            # Softmax is used, thus, muliclass segmentation
            semantic_mask = semantic_mask.argmax(dim=channel_dim, keepdim=True)
            # Create a colormap for each pixel based on the class with the highest probability
            output = colors[semantic_mask.squeeze(channel_dim)]
            if semantic_mask.dim() == 3:
                output = output.permute(2, 0, 1)
            elif semantic_mask.dim() == 4:
                output = output.permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    f"Semantic mask must be of shape (C, H, W) or (B, C, H, W), got {semantic_mask.shape}."
                )
        else:
            raise ValueError(
                "Only muliclass segmentation is supported. Please ensure a softmax is used, or submit a PR."
            )

        return output

    def visualize(
        self,
        images: Union[Tensor, list[Tensor]],
        semantic_masks: Optional[Union[Tensor, list[Tensor]]] = None,
        output_type: str = "torch",
        colormap: str = "random",
        manual_seed: int = 2147,
    ) -> Union[Tensor, list[Tensor], list[Image.Image]]:  # type: ignore
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

        outputs: Union[Tensor, list[Tensor]]
        if isinstance(
            semantic_masks,
            (
                list,
                tuple,
            ),
        ):
            outputs = []
            for semantic_mask in semantic_masks:
                if semantic_mask.ndim != 3:
                    raise ValueError(f"Semantic mask must be of shape (C, H, W), got {semantic_mask.shape}.")
                # Generate a color for each class
                colors = self.get_colormap(semantic_mask.size(0), colormap, manual_seed=manual_seed)
                outputs.append(self.visualize_output(semantic_mask, colors))

        else:
            # Generate a color for each class
            colors = self.get_colormap(semantic_masks.size(1), colormap, manual_seed=manual_seed)
            outputs = self.visualize_output(semantic_masks, colors)

        return self._tensor_to_type(outputs, output_type, is_batch=True if isinstance(outputs, Tensor) else False)

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
        overlaid: Union[Tensor, list[Tensor]]
        if isinstance(images, Tensor) and isinstance(colored_masks, Tensor):
            overlaid = kornia.enhance.add_weighted(images, 0.5, colored_masks, 0.5, 1.0)
        elif isinstance(
            images,
            (
                list,
                tuple,
            ),
        ) and isinstance(
            colored_masks,
            (
                list,
                tuple,
            ),
        ):
            overlaid = []
            for i in range(len(images)):
                overlaid.append(kornia.enhance.add_weighted(images[i][None], 0.5, colored_masks[i][None], 0.5, 1.0)[0])
        else:
            raise ValueError(f"`images` should be a Tensor or a list of Tensors. Got {type(images)}")

        self._save_outputs(images, directory, suffix="_src")
        self._save_outputs(colored_masks, directory, suffix="_mask")
        self._save_outputs(overlaid, directory, suffix="_overlay")
