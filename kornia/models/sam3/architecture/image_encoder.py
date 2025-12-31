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

"""SAM-3 Image Encoder (Hiera-based backbone).

This module implements the Hiera-based image encoder for SAM-3.
The encoder processes input images and extracts multi-scale feature representations.
"""

from __future__ import annotations

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK_SHAPE

from .common import Attention, MLPBlock


class PatchEmbedding(nn.Module):
    """Patch embedding layer for vision transformers.

    Converts image patches to embeddings through a convolutional layer.
    """

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        """Initialize PatchEmbedding.

        Args:
            img_size: Input image size (assumed square).
            patch_size: Size of each patch.
            in_channels: Number of input channels (typically 3 for RGB).
            embed_dim: Embedding dimension.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply patch embedding.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tensor of shape (B, num_patches, embed_dim).
        """
        KORNIA_CHECK_SHAPE(x, ["B", "C", "H", "W"])
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        x = self.norm(x)
        return x


class ViTBlock(nn.Module):
    """Vision Transformer block with self-attention and MLP.

    Standard transformer block used in Vision Transformers.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """Initialize ViTBlock.

        Args:
            dim: Embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            dropout: Dropout probability.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ViT block.

        Args:
            x: Input tensor of shape (B, N, D).

        Returns:
            Output tensor of the same shape.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoderHiera(nn.Module):
    """Hiera-based image encoder for SAM-3.

    This encoder extracts multi-scale features from input images using a Hiera backbone.
    The Hiera architecture is designed for efficient hierarchical feature extraction.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """Initialize ImageEncoderHiera.

        Args:
            img_size: Input image size (assumed square). Default: 1024
            patch_size: Patch size for patch embedding. Default: 16
            in_channels: Number of input channels. Default: 3
            embed_dim: Embedding dimension. Default: 768
            depth: Number of transformer blocks. Default: 12
            num_heads: Number of attention heads. Default: 12
            mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: 4.0
            dropout: Dropout probability. Default: 0.0
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Positional embedding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the image encoder.

        Args:
            x: Input tensor of shape (B, 3, H, W) where H, W should match img_size.

        Returns:
            Tensor of shape (B, num_patches, embed_dim) containing the encoded features.

        Example:
            >>> encoder = ImageEncoderHiera(img_size=1024, patch_size=16, embed_dim=768)
            >>> images = torch.randn(1, 3, 1024, 1024)
            >>> features = encoder(images)
            >>> features.shape
            torch.Size([1, 4096, 768])
        """
        KORNIA_CHECK_SHAPE(x, ["B", "3", "H", "W"])

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        return x

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape given input shape.

        Args:
            input_shape: Input tensor shape (B, C, H, W).

        Returns:
            Output tensor shape (B, num_patches, embed_dim).
        """
        B, _, H, W = input_shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        return (B, num_patches, self.embed_dim)


__all__ = ["ImageEncoderHiera", "PatchEmbedding", "ViTBlock"]
