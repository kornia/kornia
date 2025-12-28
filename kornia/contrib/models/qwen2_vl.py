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

import torch.nn.functional as F
from torch import nn

from kornia.core import Module, Tensor


# ============================================================================
# 1. The Patch Merger (The "Eyes" of the model)
# Qwen2-VL uses 3D Convolution because it treats time/video as a dimension.
# ============================================================================
class Qwen2VLPatchMerger(Module):
    def __init__(self, dim: int, context_window: int = 224, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = dim * (spatial_merge_size**2)
        self.context_window = context_window
        self.spatial_merge_size = spatial_merge_size

        # Qwen2-VL uses a 3D convolution to merge patches (Time, Height, Width)
        # Instead of HF's configuration, we hardcode the standard Qwen2-VL logic
        self.ln_q = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch_size, seq_len, dim]
        # In a full implementation, we need to reshape this based on the grid (h, w)
        # For now, we pass it through the layer norm as a placeholder for the draft
        return self.ln_q(x)


# ============================================================================
# 2. The Rotary Embedding (The "Geometry" of the model)
# This is the hardest part to export to ONNX. We define the skeleton here.
# ============================================================================
class Qwen2VLRotaryEmbedding(Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        # We will populate the inv_freq buffers later

    def forward(self, x: Tensor, seq_len: int) -> Tensor:
        return x  # Placeholder for the draft PR


# ============================================================================
# 3. The Attention Mechanism (Native PyTorch 2.0)
# Replaces HF's 'Qwen2VLVisionAttention'
# ============================================================================
class Qwen2VLVisionAttention(Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        # Flash Attention Logic (Native PyTorch)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # This is the line that makes it faster than the Alibaba implementation
        # and removes the manual einsum logic you saw in 'vit.py'
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# ============================================================================
# 4. The Main Encoder Block
# ============================================================================
class Qwen2VLVisionBlock(Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Qwen2VLVisionAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        # MLP Block
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# 5. The Main Model Class (The Entry Point)
# ============================================================================
class Qwen2VLVisionTransformer(Module):
    """Native PyTorch implementation of Qwen2-VL Vision Encoder.

    Arguments:
        embed_dim: Width of the transformer (default 1024 for Qwen2-VL-7B)
        depth: Number of layers (default 24 approx)
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        embed_dim: int = 1280,  # Default for 7B model
        depth: int = 32,
        num_heads: int = 16,
        patch_size: int = 14,
    ) -> None:
        super().__init__()
        self.patch_embed = Qwen2VLPatchMerger(embed_dim, spatial_merge_size=2)

        # The Transformer Stack
        self.blocks = nn.Sequential(*[Qwen2VLVisionBlock(embed_dim, num_heads) for _ in range(depth)])
        self.rotary_pos_emb = Qwen2VLRotaryEmbedding(embed_dim // num_heads)

    def forward(self, x: Tensor) -> Tensor:
        # x is the raw image tensor
        # 1. Patch Merge
        x = self.patch_embed(x)
        # 2. Transformer Layers
        x = self.blocks(x)
        return x
