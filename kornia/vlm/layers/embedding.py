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

"""Embedding layers for transformer models."""

import math
from typing import Optional

import torch
from torch import nn

from kornia.core import Tensor


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding.

    Converts an image into a sequence of patch embeddings using a convolution.
    Used in Vision Transformers (ViT) and similar architectures.

    Args:
        image_size: Input image size (assumes square images).
        patch_size: Size of each patch (assumes square patches).
        num_channels: Number of input channels.
        embed_dim: Dimension of the output embeddings.

    Example:
        >>> embed = PatchEmbedding(224, 14, 3, 1152)
        >>> images = torch.randn(2, 3, 224, 224)
        >>> patches = embed(images)
        >>> patches.shape
        torch.Size([2, 256, 1152])  # 224/14 = 16, 16*16 = 256 patches

    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        num_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim

        self.num_patches = (image_size // patch_size) ** 2

        # Use conv2d for efficient patch extraction and projection
        self.proj = nn.Conv2d(
            num_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Convert images to patch embeddings.

        Args:
            pixel_values: Input images of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim).

        """
        # (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(pixel_values)
        # (B, embed_dim, H', W') -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class TokenEmbedding(nn.Module):
    """Token embedding with optional positional embedding.

    Converts token IDs to embeddings and optionally adds learned positional embeddings.

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Dimension of the embeddings.
        max_position_embeddings: Maximum sequence length for positional embeddings.
            Set to 0 to disable positional embeddings.
        padding_idx: Index of the padding token.

    Example:
        >>> embed = TokenEmbedding(32000, 768, max_position_embeddings=512)
        >>> tokens = torch.randint(0, 32000, (2, 10))
        >>> embeddings = embed(tokens)
        >>> embeddings.shape
        torch.Size([2, 10, 768])

    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_position_embeddings: int = 0,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.position_embedding = None
        if max_position_embeddings > 0:
            self.position_embedding = nn.Embedding(max_position_embeddings, embed_dim)

    def forward(self, input_ids: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        """Get token embeddings with optional position embeddings.

        Args:
            input_ids: Token IDs of shape (B, seq_len).
            position_ids: Optional position indices of shape (B, seq_len).

        Returns:
            Embeddings of shape (B, seq_len, embed_dim).

        """
        embeddings = self.token_embedding(input_ids)

        if self.position_embedding is not None:
            if position_ids is None:
                seq_len = input_ids.shape[1]
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            embeddings = embeddings + self.position_embedding(position_ids)

        return embeddings


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding.

    Fixed positional embeddings using sine and cosine functions of different frequencies.
    Does not require learning and generalizes to longer sequences.

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)

    Args:
        embed_dim: Dimension of the embeddings.
        max_position_embeddings: Maximum sequence length.

    """

    def __init__(self, embed_dim: int, max_position_embeddings: int = 8192) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Create sinusoidal position embeddings
        position = torch.arange(max_position_embeddings).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pe = torch.zeros(max_position_embeddings, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Get positional embeddings for the input sequence length.

        Args:
            x: Input tensor of shape (B, seq_len, embed_dim).

        Returns:
            Positional embeddings of shape (1, seq_len, embed_dim).

        """
        seq_len = x.shape[1]
        return self.pe[:seq_len].unsqueeze(0)
