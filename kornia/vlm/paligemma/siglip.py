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

"""SigLIP Vision Encoder implementation.

SigLIP (Sigmoid Loss for Language Image Pre-training) is a vision encoder
trained with sigmoid loss instead of softmax, enabling better scalability.

Reference: https://arxiv.org/abs/2303.15343
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import Tensor

from ..base import VisionOutput
from ..layers import GeLUMLP, LayerNorm, PatchEmbedding
from .config import SigLIPVisionConfig


class SiglipSelfAttention(nn.Module):
    """Multi-head self-attention for Siglip vision encoder.

    Standard scaled dot-product attention without rotary embeddings.

    Args:
        config: Siglip vision configuration.

    """

    def __init__(self, config: SigLIPVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_size = config.head_dim
        self.scale = self.head_size**-0.5
        self.drop_rate = config.attention_dropout

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_size)
        self.wk = nn.Linear(self.dim, self.n_heads * self.head_size)
        self.wv = nn.Linear(self.dim, self.n_heads * self.head_size)
        self.wo = nn.Linear(self.n_heads * self.head_size, self.dim)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass of Siglip self-attention.

        Args:
            x: Input tensor of shape (B, seq_len, dim).
            mask: Optional attention mask.
            return_weights: Whether to return attention weights.

        Returns:
            Tuple of output tensor and optional attention weights.

        """
        B, L, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape: (B, L, n_heads * head_size) -> (B, n_heads, L, head_size)
        q = q.view(B, L, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_size).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores + mask

        # Softmax and dropout
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.training and self.drop_rate > 0:
            weights = F.dropout(weights, p=self.drop_rate)

        # Apply attention to values
        out = torch.matmul(weights, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, L, self.n_heads * self.head_size)

        # Output projection
        out = self.wo(out)

        return out, weights if return_weights else None


class SiglipTransformerBlock(nn.Module):
    """Single transformer block for Siglip vision encoder.

    Pre-norm architecture with self-attention and feedforward MLP.

    Args:
        config: Siglip vision configuration.

    """

    def __init__(self, config: SigLIPVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size

        self.attn = SiglipSelfAttention(config)
        self.norm1 = LayerNorm(self.dim, eps=config.layer_norm_eps)
        self.ffn = GeLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.norm2 = LayerNorm(self.dim, eps=config.layer_norm_eps)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass of the transformer block.

        Args:
            x: Input tensor of shape (B, seq_len, dim).
            mask: Optional attention mask.
            return_weights: Whether to return attention weights.

        Returns:
            Tuple of output tensor and optional attention weights.

        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x, attn_w = self.attn(x, mask=mask, return_weights=return_weights)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn_w


class SiglipTransformerStack(nn.Module):
    """Stack of transformer blocks for Siglip vision encoder.

    Args:
        config: Siglip vision configuration.

    """

    def __init__(self, config: SigLIPVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([SiglipTransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_weights: bool = False,
        return_intermediates: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, ...]], Optional[Tuple[Tensor, ...]]]:
        """Forward pass through all transformer blocks.

        Args:
            x: Input tensor of shape (B, seq_len, dim).
            mask: Optional attention mask.
            return_weights: Whether to return attention weights from all layers.
            return_intermediates: Whether to return features from all layers.

        Returns:
            Tuple of:
                - Final features
                - All layer features if return_intermediates=True
                - All attention weights if return_weights=True

        """
        all_features: Optional[Tuple[Tensor, ...]] = () if return_intermediates else None
        all_weights: Optional[Tuple[Tensor, ...]] = () if return_weights else None

        for block in self.blocks:
            if return_intermediates:
                all_features = all_features + (x,)

            x, attn_w = block(x, mask=mask, return_weights=return_weights)

            if return_weights and attn_w is not None:
                all_weights = all_weights + (attn_w,)

        # Add final features
        if return_intermediates:
            all_features = all_features + (x,)

        return x, all_features, all_weights


class SiglipPatchEmbedder(nn.Module):
    """Patch embedding layer for Siglip vision encoder.

    Combines patch embeddings with learnable position embeddings.

    Args:
        config: Siglip vision configuration.

    """

    def __init__(self, config: SigLIPVisionConfig) -> None:
        super().__init__()
        self.config = config

        self.patch_proj = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )

        self.n_patches = self.patch_proj.num_patches
        self.pos_embed = nn.Embedding(self.n_patches, config.hidden_size)

        # Register position indices buffer for efficiency
        self.register_buffer(
            "pos_indices",
            torch.arange(self.n_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, images: Tensor) -> Tensor:
        """Get patch + position embeddings.

        Args:
            images: Input images of shape (B, C, H, W).

        Returns:
            Embeddings of shape (B, n_patches, dim).

        """
        patches = self.patch_proj(images)
        embeddings = patches + self.pos_embed(self.pos_indices)
        return embeddings


class SiglipVisionEncoder(nn.Module):
    """Siglip Vision Encoder Model.

    Complete vision encoder that can be used standalone for feature extraction
    or as part of a vision-language model.

    Args:
        config: Siglip vision configuration.

    Example:
        >>> config = SigLIPVisionConfig()
        >>> encoder = SiglipVisionEncoder(config)
        >>> images = torch.randn(2, 3, 224, 224)
        >>> output = encoder(images)
        >>> output.features.shape
        torch.Size([2, 256, 1152])

    """

    def __init__(self, config: SigLIPVisionConfig) -> None:
        super().__init__()
        self.config = config

        self.embedder = SiglipPatchEmbedder(config)
        self.transformer = SiglipTransformerStack(config)
        self.final_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @property
    def num_patches(self) -> int:
        """Number of patches in the image."""
        return self.embedder.n_patches

    def forward(
        self,
        images: Tensor,
        return_attention_weights: bool = False,
        return_intermediates: bool = False,
    ) -> VisionOutput:
        """Forward pass of the Siglip vision encoder.

        Args:
            images: Input images of shape (B, C, H, W).
            return_attention_weights: Whether to return attention weights from all layers.
            return_intermediates: Whether to return features from all layers.

        Returns:
            VisionOutput with features and optional intermediates.

        """
        # Get embeddings
        x = self.embedder(images)

        # Pass through transformer
        x, layer_features, attn_weights = self.transformer(
            x,
            return_weights=return_attention_weights,
            return_intermediates=return_intermediates,
        )

        # Final normalization
        features = self.final_norm(x)

        return VisionOutput(
            features=features,
            layer_features=layer_features,
            attention_weights=attn_weights,
        )

    @classmethod
    def from_config(cls, config: SigLIPVisionConfig) -> "SiglipVisionEncoder":
        """Create a model from configuration.

        Args:
            config: Siglip vision configuration.

        Returns:
            Instantiated model.

        """
        return cls(config)


# Alias for backward compatibility
SigLIPVisionModel = SiglipVisionEncoder
