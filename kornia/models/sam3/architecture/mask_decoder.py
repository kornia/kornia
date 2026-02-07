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

"""SAM-3 Mask Decoder for generating segmentation masks.

This module implements the mask decoder for SAM-3 which takes image embeddings
and prompt embeddings to generate segmentation masks and IoU predictions.
"""

from __future__ import annotations

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK

from .common import Attention, MLPBlock


class CrossAttentionTransformer(nn.Module):
    """Transformer block with cross-attention between image and prompt embeddings.

    Applies cross-attention from prompt embeddings to image embeddings, followed by self-attention.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0) -> None:
        """Initialize CrossAttentionTransformer.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.self_attn = Attention(embed_dim, heads=num_heads)
        self.norm3 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLPBlock(embed_dim, mlp_dim)

    def forward(
        self,
        prompts: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention transformer.

        Args:
            prompts: Prompt embeddings of shape (B, M, D).
            image_embeddings: Image embeddings of shape (B, N, D).

        Returns:
            Updated prompt embeddings of shape (B, M, D).
        """
        # Cross-attention: prompts attend to image embeddings
        prompts_norm = self.norm1(prompts)
        attn_out, _ = self.cross_attn(prompts_norm, image_embeddings, image_embeddings)
        prompts = prompts + attn_out

        # Self-attention on prompts
        prompts_norm = self.norm2(prompts)
        self_attn_out = self.self_attn(prompts_norm)
        prompts = prompts + self_attn_out

        # MLP
        prompts_norm = self.norm3(prompts)
        mlp_out = self.mlp(prompts_norm)
        prompts = prompts + mlp_out

        return prompts


class MaskDecoder(nn.Module):
    """Mask decoder for SAM-3 that generates segmentation masks.

    Takes image embeddings and prompt embeddings to produce segmentation masks
    and IoU predictions for the prompted regions.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_multimask_outputs: int = 3,
        activation: str = "gelu",
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """Initialize MaskDecoder.

        Args:
            embed_dim: Embedding dimension.
            num_multimask_outputs: Number of mask outputs per prompt.
            activation: Activation function name.
            iou_head_depth: Depth of IoU prediction head.
            iou_head_hidden_dim: Hidden dimension of IoU head.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.activation = activation

        # Transformer for processing prompts
        self.transformer = CrossAttentionTransformer(embed_dim)

        # Mask tokens (Phase 3: added for multi-mask generation)
        self.mask_tokens = nn.ParameterList(
            [nn.Parameter(torch.randn(1, 1, embed_dim)) for _ in range(num_multimask_outputs)]
        )

        # Hypernetwork MLPs for mask generation (Phase 3: added for multi-mask support)
        self.mask_mlps = nn.ModuleList([MLPBlock(embed_dim, embed_dim * 4) for _ in range(num_multimask_outputs)])

        # Per-mask dynamic projection/linear layers for per-channel modulation
        # Each mask gets its own linear projection to modulate feature channels
        self.mask_feature_modulators = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim // 8) for _ in range(num_multimask_outputs)]
        )

        # Dense prompt processor: projects dense embeddings to per-channel modulation
        # This ensures dense_prompt_embeddings have a spatial effect on each feature channel
        self.dense_to_feature_modulation = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # Mask prediction head with hypernetwork outputs
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, embed_dim // 4),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
        )

        # Mask prediction projection head (Phase 3: one per output mask)
        self.mask_prediction_heads = nn.ModuleList(
            [nn.Conv2d(embed_dim // 8, 1, kernel_size=1) for _ in range(num_multimask_outputs)]
        )

        # IoU prediction head
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(embed_dim, iou_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(iou_head_hidden_dim, iou_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(iou_head_hidden_dim, num_multimask_outputs),
        )

    def _predict_masks(
        self,
        image_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict masks from embeddings.

        Args:
            image_embeddings: Image embeddings of shape (B, N, D).
            sparse_prompt_embeddings: Sparse prompt embeddings of shape (B, M, D).
            dense_prompt_embeddings: Dense prompt embeddings of shape (B, D, H, W).
            multimask_output: If True, generate multiple masks. Otherwise, generate single mask.

        Returns:
            Tuple of (masks, iou_pred) where:
                - masks: Tensor of shape (B, num_masks, H, W).
                - iou_pred: Tensor of shape (B, num_masks).
        """
        B, N, _ = image_embeddings.shape

        # Infer spatial dimensions from sequence length
        # image_embeddings: (B, N, D) where N = H*W for square grid (per batch)
        H = W = int(N**0.5)
        KORNIA_CHECK(H * W == N, f"image_embeddings must form a square grid. Got N={N}")

        # Reshape image embeddings to spatial form for processing
        image_embeddings_spatial = image_embeddings.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)

        # Process dense prompts to create per-channel modulation
        # Apply modulation to ensure dense_prompt_embeddings have spatial effect
        if dense_prompt_embeddings.shape[1] > 0:
            # Project dense prompts to match spatial size of image embeddings
            dense_resized = torch.nn.functional.interpolate(
                dense_prompt_embeddings,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )  # (B, embed_dim, H, W)

            # Apply per-channel modulation from dense prompts
            # This ensures dense_prompt_embeddings genuinely affect feature channels
            dense_modulation = torch.nn.functional.relu(self.dense_to_feature_modulation(dense_resized))

            # Additive conditioning on image embeddings
            image_embeddings_spatial = image_embeddings_spatial + dense_modulation

        # Reshape back to sequence form for transformer processing
        image_with_prompts = image_embeddings_spatial.permute(0, 2, 3, 1).reshape(B, H * W, self.embed_dim)

        # Process sparse prompts through transformer (feeds prompts into transformer decoder)
        if sparse_prompt_embeddings.shape[1] > 0:
            sparse_processed = self.transformer(sparse_prompt_embeddings, image_with_prompts)
        else:
            sparse_processed = sparse_prompt_embeddings

        # Get primary prompt representation for IoU and multi-mask decisions
        if sparse_prompt_embeddings.shape[1] > 0:
            iou_input = sparse_processed.mean(dim=1)  # (B, D)
        else:
            iou_input = torch.zeros(B, self.embed_dim, device=image_embeddings.device, dtype=image_embeddings.dtype)

        # Upscale feature map for mask prediction
        upscaled_features = self.output_upscaling(image_embeddings_spatial)  # (B, D/8, H_out, W_out)

        # Phase 3: Generate multiple masks using mask tokens and per-mask channel scaling
        num_masks_to_generate = self.num_multimask_outputs if multimask_output else 1

        masks_list = []
        for mask_idx in range(num_masks_to_generate):
            # Get mask token for this output
            mask_token = self.mask_tokens[mask_idx]  # (1, 1, D)

            # Apply hypernetwork MLP to modulate the prompt representation
            prompt_modulation = self.mask_mlps[mask_idx](iou_input)  # (B, D)

            # Combine mask token with modulated prompt for multi-mask generation
            combined = mask_token + prompt_modulation.unsqueeze(1)  # (B, 1, D)

            # Get per-mask channel scaling via linear projection
            # Different mask tokens → genuinely different per-channel modulation
            mask_channel_scale = self.mask_feature_modulators[mask_idx](combined.squeeze(1))  # (B, D/8)

            # Apply mask prediction head
            mask_logits = self.mask_prediction_heads[mask_idx](upscaled_features)  # (B, 1, H_out, W_out)

            # Apply per-channel modulation to mask logits
            # Reshape for broadcasting: (B, D/8) -> (B, D/8, 1, 1)
            mask_channel_scale = torch.nn.functional.relu(mask_channel_scale)
            mask_channel_scale = mask_channel_scale.view(B, -1, 1, 1)

            # Modulate the upscaled features before final mask prediction
            modulated_features = upscaled_features * mask_channel_scale
            mask_logits = self.mask_prediction_heads[mask_idx](modulated_features)  # (B, 1, H_out, W_out)

            masks_list.append(mask_logits)

        # Concatenate all masks
        masks = torch.cat(masks_list, dim=1)  # (B, num_masks_to_generate, H_out, W_out)

        # Predict IoU scores matching the number of generated masks
        iou_pred_all = self.iou_prediction_head(iou_input)  # (B, num_multimask_outputs)
        iou_pred = iou_pred_all[:, :num_masks_to_generate]  # (B, num_masks_to_generate)

        return masks, iou_pred

    def forward(
        self,
        image_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate segmentation masks from image and prompt embeddings.

        Args:
            image_embeddings: Image embeddings of shape (B, N, D) from image encoder.
            sparse_prompt_embeddings: Sparse prompt embeddings of shape (B, M, D) from prompt encoder.
            dense_prompt_embeddings: Dense prompt embeddings of shape (B, D, H, W) from prompt encoder.
            multimask_output: If True, output multiple masks per prompt. Otherwise, output single mask.

        Returns:
            Tuple of (masks, iou_pred) where:
                - masks: Segmentation masks of shape (B, num_masks, H, W).
                - iou_pred: IoU predictions of shape (B, num_masks).

        Raises:
            ValueError: If image_embeddings shape is invalid.
        """
        KORNIA_CHECK(
            image_embeddings.ndim == 3,
            f"image_embeddings must be 3D (B, N, D), got shape {image_embeddings.shape}",
        )
        KORNIA_CHECK(
            sparse_prompt_embeddings.ndim == 3,
            f"sparse_prompt_embeddings must be 3D (B, M, D), got shape {sparse_prompt_embeddings.shape}",
        )
        KORNIA_CHECK(
            dense_prompt_embeddings.ndim == 4,
            f"dense_prompt_embeddings must be 4D (B, D, H, W), got shape {dense_prompt_embeddings.shape}",
        )

        # Predict masks
        masks, iou_pred = self._predict_masks(
            image_embeddings,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            multimask_output=multimask_output,
        )

        # Phase 3: multimask_output parameter is now respected and meaningful
        # Generate multiple masks when multimask_output=True, single mask when False

        return masks, iou_pred


__all__ = ["CrossAttentionTransformer", "MaskDecoder"]
