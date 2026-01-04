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
from torch import Tensor, nn
from torch.nn import Module


class Qwen2VLPatchMerger(Module):
    """Patch merger block used in the Qwen2-VL vision encoder.

    Args:
        dim: The output embedding dimension (e.g., 1280).
        context_window: The context window size (unused in this skeleton but kept for API).
        spatial_merge_size: The spatial merge size (unused in this skeleton).
    """

    def __init__(self, dim: int, context_window: int = 224, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, dim, kernel_size=14, stride=14)
        self.ln_q = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.ln_q(x)
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

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, cu_seqlens: Optional[Tensor] = None) -> Tensor:
        return x


class Qwen2VLVisionAttention(Module):
    """Multi-head self-attention module used in the Qwen2-VL vision encoder.

    Args:
        dim: Input feature dimension.
        num_heads: Number of attention heads.
    """

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
    """Single transformer block for the Qwen2-VL vision encoder.

    Applies layer-normalized self-attention followed by an MLP, each with
    residual connections.

    Args:
        dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension multiplier.
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
    """PyTorch implementation of the Qwen2-VL vision encoder.

    A ViT-style backbone composed of a patch merger followed by stacked
    transformer blocks with rotary positional embeddings.

    Args:
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        patch_size: Patch size.
        in_channels: Input channel count.
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_embed = Qwen2VLPatchMerger(embed_dim, context_window=224, spatial_merge_size=2)
        self.blocks = nn.ModuleList([Qwen2VLVisionBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.rotary_pos_emb = Qwen2VLRotaryEmbedding(embed_dim // num_heads)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        rot_pos_emb = self.rotary_pos_emb(x)
        for block in self.blocks:
            x = block(x, rot_pos_emb=rot_pos_emb)

        return x
