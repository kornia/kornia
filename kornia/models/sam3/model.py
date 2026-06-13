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

"""SAM-3 Model wrapper for end-to-end segmentation.

This module provides the high-level Sam3 model that combines the image encoder,
prompt encoder, and mask decoder for complete segmentation inference.
"""

from __future__ import annotations

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE

from .architecture import ImageEncoderHiera, MaskDecoder, PromptEncoder


class Sam3(nn.Module):
    """SAM-3 (Segment Anything Model v3) end-to-end model.

    Combines image encoder, prompt encoder, and mask decoder for image segmentation.
    Supports point, box, and mask prompts for flexible segmentation control.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        encoder_depth: int = 12,
        num_heads: int = 12,
        num_multimask_outputs: int = 3,
        mask_in_chans: int = 16,
    ) -> None:
        """Initialize Sam3 model.

        Args:
            img_size: Input image size (assumed square). Default: 1024
            patch_size: Patch size for image encoder. Default: 16
            in_channels: Number of input image channels. Default: 3
            embed_dim: Embedding dimension for all components. Default: 256
            encoder_depth: Number of transformer blocks in image encoder. Default: 12
            num_heads: Number of attention heads. Default: 12
            num_multimask_outputs: Number of mask outputs per prompt. Default: 3
            mask_in_chans: Number of input channels for mask encoding. Default: 16
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Image encoder (Phase 1)
        self.image_encoder = ImageEncoderHiera(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
        )

        # Prompt encoder (Phase 2/3)
        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            input_image_size=img_size,
            mask_in_chans=mask_in_chans,
        )

        # Mask decoder (Phase 2/3)
        self.mask_decoder = MaskDecoder(
            embed_dim=embed_dim,
            num_multimask_outputs=num_multimask_outputs,
        )

    def forward(
        self,
        images: torch.Tensor,
        *,
        points: tuple[torch.Tensor, torch.Tensor] | None = None,
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        multimask_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate segmentation masks for prompted regions.

        Args:
            images: Input images of shape (B, 3, H, W) where H, W match img_size.
            points: Optional tuple of (coords, labels) for point prompts.
                - coords: Shape (B, N, 2) with normalized coordinates in [0, 1].
                - labels: Shape (B, N) with binary labels (0=background, 1=foreground).
            boxes: Optional tensor of shape (B, num_boxes, 4) with normalized bbox coordinates
                in format [x_min, y_min, x_max, y_max].
            masks: Optional tensor of shape (B, 1, H, W) with binary mask inputs.
            multimask_output: If True, output multiple masks per prompt. Otherwise, single mask.

        Returns:
            Tuple of (masks, iou_pred) where:
                - masks: Segmentation masks of shape (B, num_masks, H_out, W_out).
                - iou_pred: IoU predictions of shape (B, num_masks).

        Raises:
            ValueError: If input shapes are invalid.
        """
        KORNIA_CHECK_SHAPE(images, ["B", str(self.image_encoder.patch_embed.proj.in_channels), "H", "W"])
        KORNIA_CHECK(
            images.shape[2] == self.img_size and images.shape[3] == self.img_size,
            f"Input images must be {self.img_size}x{self.img_size}, got {images.shape[2]}x{images.shape[3]}",
        )

        # Phase 1: Encode images
        image_embeddings = self.image_encoder(images)  # (B, num_patches, embed_dim)

        # Phase 2/3: Encode prompts
        sparse_prompt_embeddings, dense_prompt_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )

        # Phase 2/3: Decode masks
        masks_out, iou_pred = self.mask_decoder(
            image_embeddings,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            multimask_output=multimask_output,
        )

        return masks_out, iou_pred


__all__ = ["Sam3"]
