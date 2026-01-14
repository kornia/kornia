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

"""SAM-3 model wrapper combining image encoder, prompt encoder, and mask decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK_SHAPE

from .architecture import ImageEncoderHiera, MaskDecoder, PromptEncoder
from .config import Sam3Config, Sam3ModelType


@dataclass
class SegmentationResults:
    """Segmentation results from SAM-3.

    Attributes:
        masks: Segmentation masks of shape (B, C, H, W).
        logits: Mask logits of shape (B, C, H, W).
        iou_pred: Predicted IoU scores of shape (B, C).
    """

    masks: torch.Tensor
    logits: torch.Tensor
    iou_pred: torch.Tensor


class Sam3(nn.Module):
    """SAM-3 model for image segmentation.

    Combines image encoder, prompt encoder, and mask decoder for end-to-end segmentation.
    """

    def __init__(
        self,
        config: Sam3Config | None = None,
    ) -> None:
        """Initialize SAM-3 model.

        Args:
            config: Sam3Config for architecture parameters. If None, uses default base config.
        """
        super().__init__()
        if config is None:
            config = Sam3Config(model_type=Sam3ModelType.base)

        self.config = config

        # Image encoder
        self.image_encoder = ImageEncoderHiera(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.encoder_embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_num_heads,
            mlp_ratio=config.encoder_mlp_ratio,
        )

        # Prompt encoder
        self.prompt_encoder = PromptEncoder(
            embed_dim=config.prompt_embed_dim,
            input_image_size=config.img_size,
            mask_in_chans=config.mask_in_chans,
        )

        # Mask decoder
        self.mask_decoder = MaskDecoder(
            embed_dim=config.decoder_embed_dim,
            num_multimask_outputs=config.num_multimask_outputs,
            iou_head_depth=config.iou_head_depth,
            iou_head_hidden_dim=config.iou_head_hidden_dim,
        )

        # Projection from image encoder to decoder embedding space if needed
        if config.encoder_embed_dim != config.decoder_embed_dim:
            self.image_embedding_projection = nn.Linear(config.encoder_embed_dim, config.decoder_embed_dim, bias=False)
        else:
            self.image_embedding_projection = nn.Identity()

    @classmethod
    def from_config(cls, config: Sam3Config | None = None) -> Sam3:
        """Create SAM-3 model from configuration.

        Args:
            config: Sam3Config instance.

        Returns:
            Sam3 model instance.
        """
        return cls(config)

    def forward(
        self,
        images: torch.Tensor,
        points: tuple[torch.Tensor, torch.Tensor] | None = None,
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        multimask_output: bool = True,
    ) -> SegmentationResults:
        """Generate segmentation masks for input images and prompts.

        Args:
            images: Input images of shape (B, 3, H, W) where H, W = img_size.
            points: Optional point prompts as tuple of (coords, labels).
                - coords: (B, N, 2) normalized coordinates in [0, 1].
                - labels: (B, N) with 0=background, 1=foreground.
            boxes: Optional bounding box prompts of shape (B, num_boxes, 4) with (x0, y0, x1, y1).
            masks: Optional mask prompts of shape (B, 1, H, W) with binary masks.
            multimask_output: If True, return all masks. Otherwise, return best mask.

        Returns:
            SegmentationResults with masks, logits, and IoU predictions.

        Raises:
            ValueError: If image shape is invalid.
        """
        _, _, H, W = images.shape
        KORNIA_CHECK_SHAPE(images, ["B", "3", str(H), str(W)])

        # Image encoding
        image_embeddings = self.image_encoder(images)  # (B, num_patches, encoder_embed_dim)

        # Project to decoder embedding space if needed
        image_embeddings = self.image_embedding_projection(image_embeddings)  # (B, num_patches, decoder_embed_dim)

        # Prompt encoding
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )

        # Mask decoding
        mask_logits, iou_pred = self.mask_decoder(
            image_embeddings=image_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Convert logits to probabilities
        mask_probs = torch.sigmoid(mask_logits)

        return SegmentationResults(
            masks=mask_probs,
            logits=mask_logits,
            iou_pred=iou_pred,
        )


__all__ = ["Sam3", "SegmentationResults"]
