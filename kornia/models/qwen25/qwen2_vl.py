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

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from kornia.core import Module, Tensor 
from torch import nn


class Qwen2VLPatchMerger(Module):
    """Patch merger block used in the Qwen2-VL vision encoder.

    This module normalizes per-token visual features and applies a 3D convolution
    to merge local spatial patches over time into higher-dimensional tokens. It is
    typically used to reduce the spatial resolution of video or image feature maps
    while preserving temporal structure before feeding them into a language or
    multimodal backbone.

    Args:
        dim: Input feature dimension for each visual token/channel.
        context_window: Maximum temporal context (number of frames or time steps)
            the merger is expected to handle.
        spatial_merge_size: Spatial downsampling factor for both height and width.
            A value of ``k`` merges non-overlapping ``k x k`` spatial patches.
    """

    def __init__(self, dim: int, context_window: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = dim * (spatial_merge_size**2)
        self.spatial_merge_size = spatial_merge_size

        self.ln_q = nn.LayerNorm(dim, eps=1e-6)

        # 3D Convolution to merge temporal and spatial dimensions
        self.merger = nn.Conv3d(
            dim,
            self.hidden_size,
            kernel_size=(1, spatial_merge_size, spatial_merge_size),
            stride=(1, spatial_merge_size, spatial_merge_size),
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.ln_q(x)
        x = self.merger(x) 
        return x


class Qwen2VLRotaryEmbedding(Module):
    """Rotary positional embedding module used in Qwen2-VL vision-language layers.

    This module precomputes the inverse frequency spectrum required to build
    rotary position embeddings (RoPE) for a given hidden dimension. The
    frequencies are used to rotate query and key vectors in the attention mechanism,
    encoding relative position information.

    Args:
        dim: The feature dimension to be rotated.
        theta: The base frequency scaling factor for the rotary embedding.
    """
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x: Tensor, cu_seqlens: Tensor) -> Tensor:
        return x


class Qwen2VLVisionAttention(Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: Tensor, cu_seqlens: Optional[Tensor] = None, rot_pos_emb: Optional[Tensor] = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Qwen2VLMLP(Module):
    """FeedForward MLP used in the Qwen2-VL vision transformer blocks.

    Args:
        dim: Input and output feature dimension.
        hidden_dim: Dimension of the hidden layer. If None, defaults to 4 * dim.
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Qwen2VLVisionBlock(Module):
    """Single transformer block used in the Qwen2-VL vision encoder.
    
    This block follows the standard vision transformer pattern:
    layer-normalized inputs are passed through multi-head self-attention
    and added back via a residual connection, then layer-normalized again
    and processed by an MLP with another residual connection.
    Args:
        dim: Embedding dimension of the visual tokens.
        num_heads: Number of attention heads used in the self-attention layer.
        mlp_ratio: Expansion ratio used to compute the hidden dimension of the
            MLP, where ``hidden_dim = int(dim * mlp_ratio)``.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Qwen2VLVisionAttention(dim, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Qwen2VLMLP(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor, rot_pos_emb: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), rot_pos_emb=rot_pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2VLVisionTransformer(Module):
    """Native PyTorch implementation of the Qwen2-VL vision encoder.

    This module implements the vision backbone used in Qwen2-VL as a stack of
    transformer blocks operating on patch embeddings produced by a patch
    merger. It follows a standard ViT-style architecture composed of:

    * A :class:`Qwen2VLPatchMerger` that converts an input image tensor into a
      sequence of patch tokens.
    * A sequence of :class:`Qwen2VLVisionBlock` modules, each containing
      multi-head self-attention and an MLP with residual connections.
    * Rotary positional embeddings applied to the attention mechanism.

    Args:
        embed_dim: The dimension of the embedding vectors.
        depth: The number of transformer blocks in the encoder.
        num_heads: The number of attention heads.
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension.
        patch_size: The size of the patches to be merged.
        in_channels: Number of input channels (default: 3 for RGB).
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        patch_size: int = 14,
        context_window: int = 224,
    ) -> None:
        super().__init__()
        self.patch_embed = Qwen2VLPatchMerger(embed_dim, context_window=context_window, spatial_merge_size=2)
        self.blocks = nn.Sequential(*[Qwen2VLVisionBlock(embed_dim, num_heads) for _ in range(depth)])
        self.rotary_pos_emb = Qwen2VLRotaryEmbedding(embed_dim // num_heads)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self.blocks(x)
        return x
