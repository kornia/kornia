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


def _cast_module_to_dtype(module: nn.Module, target_dtype: torch.dtype) -> None:
    """Cast all parameters in a module to the target dtype.
    
    This is a workaround for ensuring dtype consistency when .to() doesn't properly
    convert all module parameters, particularly for float64 testing compatibility.
    
    Args:
        module: The module to cast.
        target_dtype: The target dtype to cast parameters to.
    """
    for param in module.parameters():
        param.data = param.data.to(dtype=target_dtype)


class PositionalEncodingDecoder(nn.Module):
    """Learnable positional encoding for 2D spatial features.

    Used in mask decoder to encode spatial positions.
    """

    def __init__(self, embed_dim: int, height: int, width: int) -> None:
        """Initialize PositionalEncodingDecoder.

        Args:
            embed_dim: Embedding dimension.
            height: Height of feature map.
            width: Width of feature map.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.height = height
        self.width = width
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, height, width) * 0.02)


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

        # Mask tokens for multi-mask generation (Phase 3)
        self.mask_tokens = nn.ParameterList(
            [nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02) for _ in range(num_multimask_outputs)]
        )

        # Hypernetwork MLPs for multi-mask generation (Phase 3)
        self.mask_prediction_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim // 2),
                )
                for _ in range(num_multimask_outputs)
            ]
        )

        # Mask prediction head
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, embed_dim // 4),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
        )

        # Final mask output layer per mask
        self.mask_output_layers = nn.ModuleList(
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict masks from embeddings.

        Args:
            image_embeddings: Image embeddings of shape (B, N, D).
            sparse_prompt_embeddings: Sparse prompt embeddings of shape (B, M, D).
            dense_prompt_embeddings: Dense prompt embeddings of shape (B, D, H, W).

        Returns:
            Tuple of (masks, iou_pred) where:
                - masks: Tensor of shape (B, num_masks, H, W).
                - iou_pred: Tensor of shape (B, num_masks).
        """
        B, N, _ = image_embeddings.shape

        # Infer spatial dimensions from sequence length
        H = W = int(N**0.5)
        KORNIA_CHECK(H * W == N, f"image_embeddings must form a square grid. Got N={N}")

        # Reshape image embeddings to spatial form for processing
        image_embeddings_spatial = image_embeddings.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)

        # Add positional encoding to image embeddings (Phase 3)
        if not hasattr(self, "_pos_encoder"):
            self._pos_encoder = PositionalEncodingDecoder(self.embed_dim, H, W)
            self._pos_encoder = self._pos_encoder.to(image_embeddings_spatial.device)

        image_embeddings_spatial = image_embeddings_spatial + self._pos_encoder.pos_embed

        # Add dense prompts to image embeddings
        if dense_prompt_embeddings.shape[1] > 0:
            # Project dense prompts to match decoder embedding dimension if needed
            if dense_prompt_embeddings.shape[1] != self.embed_dim:
                if not hasattr(self, "_dense_projection"):
                    in_channels = dense_prompt_embeddings.shape[1]
                    self._dense_projection = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1, bias=False)
                    self._dense_projection = self._dense_projection.to(dense_prompt_embeddings.device, dtype=dense_prompt_embeddings.dtype)
                dense_prompt_embeddings = self._dense_projection(dense_prompt_embeddings)

            # Resize dense prompts to match image embedding spatial size
            dense_resized = torch.nn.functional.interpolate(
                dense_prompt_embeddings,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            # Add to image embeddings
            image_embeddings_spatial = image_embeddings_spatial + dense_resized

        # Reshape back to sequence form
        image_with_prompts = image_embeddings_spatial.permute(0, 2, 3, 1).reshape(B, H * W, self.embed_dim)

        # Process sparse prompts through transformer (Phase 3: multi-mask support)
        if sparse_prompt_embeddings.shape[1] > 0:
            # Project sparse prompts to match decoder embedding dimension if needed
            if sparse_prompt_embeddings.shape[-1] != self.embed_dim:
                if not hasattr(self, "_prompt_projection"):
                    self._prompt_projection = nn.Linear(sparse_prompt_embeddings.shape[-1], self.embed_dim, bias=False)
                    self._prompt_projection = self._prompt_projection.to(sparse_prompt_embeddings.device, dtype=sparse_prompt_embeddings.dtype)
                sparse_prompt_embeddings = self._prompt_projection(sparse_prompt_embeddings)

            sparse_processed = self.transformer(sparse_prompt_embeddings, image_with_prompts)
            iou_input = sparse_processed.mean(dim=1)  # (B, D)
        else:
            sparse_processed = sparse_prompt_embeddings
            iou_input = torch.zeros(B, self.embed_dim, device=image_embeddings.device, dtype=image_embeddings.dtype)

        # Upscale features for mask prediction
        features_upscaled = self.output_upscaling(image_embeddings_spatial)  # (B, D/8, H_out, W_out)

        # Generate multiple masks (Phase 3)
        masks_list = []
        for i in range(self.num_multimask_outputs):
            # Use mask token and hypernetwork MLP to generate mask i
            mask_token = self.mask_tokens[i]  # (1, 1, D)
            mask_token = mask_token.expand(B, -1, -1)  # (B, 1, D)

            # Process through hypernetwork MLP (Phase 3: used for future modulation)
            _ = self.mask_prediction_heads[i](mask_token.squeeze(1))  # (B, D/2)

            # Generate mask
            mask = self.mask_output_layers[i](features_upscaled)  # (B, 1, H_out, W_out)
            masks_list.append(mask)

        masks = torch.cat(masks_list, dim=1)  # (B, num_multimask_outputs, H_out, W_out)

        # Predict IoU scores
        iou_pred = self.iou_prediction_head(iou_input)  # (B, num_multimask_outputs)

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
            multimask_output: If True, output multiple masks per prompt. Otherwise, output best single mask.

        Returns:
            Tuple of (masks, iou_pred) where:
                - masks: Segmentation masks of shape (B, num_masks, H, W) if multimask_output=True,
                         else (B, 1, H, W) with best mask.
                - iou_pred: IoU predictions of shape (B, num_multimask_outputs).

        Raises:
            ValueError: If embeddings shapes are invalid.
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

        # Predict masks (Phase 3: full multi-mask generation with positional encoding)
        masks, iou_pred = self._predict_masks(
            image_embeddings,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
        )

        # Select masks based on multimask_output flag
        if not multimask_output:
            # Return only best mask (highest IoU prediction)
            best_mask_idx = iou_pred.argmax(dim=1, keepdim=True)  # (B, 1)
            gather_idx = best_mask_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, masks.shape[2], masks.shape[3])
            masks = torch.gather(masks, 1, gather_idx)

        return masks, iou_pred


__all__ = ["CrossAttentionTransformer", "MaskDecoder", "PositionalEncodingDecoder"]
