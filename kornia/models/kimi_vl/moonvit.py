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

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .config import MoonViTConfig


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings.

    Args:
        x: Input tensor of shape (batch, seq_len, head_dim).
        cos: Cosine component of shape (seq_len, head_dim).
        sin: Sine component of shape (seq_len, head_dim).

    Returns:
        Tensor with rotary embeddings applied.
    """
    # x: (batch, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim) or (1, seq_len, head_dim)

    # rotate_half
    x1, x2 = x.chunk(2, dim=-1)
    x_rotated = torch.cat((-x2, x1), dim=-1)

    return (x * cos) + (x_rotated * sin)


class MoonViTRotaryEmbedding(nn.Module):
    """2D Rotary Positional Embedding."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # dim must be divisible by 2 for 2D RoPE (half for H, half for W)
        # And each half must be divisible by 2 for complex rotation
        dim_h = self.dim // 2
        dim_w = self.dim // 2

        # Generate frequencies
        inv_freq_h = 1.0 / (self.theta ** (torch.arange(0, dim_h, 2, device=device).float() / dim_h))
        inv_freq_w = 1.0 / (self.theta ** (torch.arange(0, dim_w, 2, device=device).float() / dim_w))

        # Generate positions
        seq_h = torch.arange(h, device=device, dtype=inv_freq_h.dtype)
        seq_w = torch.arange(w, device=device, dtype=inv_freq_w.dtype)

        # Outer product to get (h, dim_h/2) and (w, dim_w/2)
        freqs_h = torch.outer(seq_h, inv_freq_h)  # (h, dim_h/2)
        freqs_w = torch.outer(seq_w, inv_freq_w)  # (w, dim_w/2)

        # Repeat h frequencies for each w
        freqs_h = freqs_h.repeat_interleave(w, dim=0)  # (h*w, dim_h/2)

        # Repeat w frequencies for each h
        freqs_w = freqs_w.repeat(h, 1)  # (h*w, dim_w/2)

        # Concatenate to get full embeddings
        emb_h = torch.cat((freqs_h, freqs_h), dim=-1)  # (seq_len, dim_h)
        emb_w = torch.cat((freqs_w, freqs_w), dim=-1)  # (seq_len, dim_w)

        emb = torch.cat((emb_h, emb_w), dim=-1)  # (seq_len, dim)

        return emb.cos(), emb.sin()


class MoonViTAttention(nn.Module):
    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape query, key, and value for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        # cos, sin are (seq_len, head_dim) -> reshape to (1, 1, seq_len, head_dim)
        cos = cos.view(1, 1, seq_len, self.head_dim)
        sin = sin.view(1, 1, seq_len, self.head_dim)

        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=self.dropout.p if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attn_output)


class MoonViTMLP(nn.Module):
    """Feed-forward MLP block used in MoonViT transformer layers.

    This module implements a two-layer projection with GELU activation and dropout,
    following the standard transformer feed-forward network pattern.

    Args:
        config: Model configuration containing ``hidden_size``, ``intermediate_size``,
            and ``dropout_p`` used to construct the MLP.
    """
    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MoonViTLayer(nn.Module):
    """Single MoonViT transformer layer with pre-normalization, RoPE attention, and an MLP block.

    This layer applies layer normalization before both the self-attention and MLP submodules
    (pre-norm transformer). The self-attention block uses rotary positional embeddings (RoPE)
    via the provided cosine and sine tensors, and the output of each sub-block is added back
    to the input (residual connections).

    Args:
        config: Model configuration specifying hidden sizes, number of heads, dropout, and
            normalization parameters.
    """
    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = MoonViTAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MoonViTMLP(config)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MoonViTEncoder(nn.Module):
    """Stack of MoonViT transformer layers.

    This encoder sequentially applies multiple :class:`MoonViTLayer` blocks to a
    sequence of hidden states. Each layer consists of self-attention with rotary
    positional embeddings followed by an MLP block, with residual connections and
    layer normalization, producing an encoded representation of the input sequence.
    """
    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MoonViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)
        return x


class MoonViT(nn.Module):
    """MoonViT Vision Encoder."""

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.config = config

        self.patch_embed = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )

        # Initialized for the default image size
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, config.hidden_size))

        self.rope = MoonViTRotaryEmbedding(config.hidden_size // config.num_attention_heads, theta=config.rope_theta)

        self.encoder = MoonViTEncoder(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: (B, C, H, W)
            attention_mask: (B, 1, N, N) or (B, N, N) optional mask.

        Returns:
            (B, seq_len, hidden_size)
        """
        _B, _C, _H, _W = pixel_values.shape

        # Patch Embedding
        x = self.patch_embed(pixel_values)  # (B, D, H', W')
        h_patches = x.shape[2]
        w_patches = x.shape[3]

        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add Absolute Positional Embedding (with interpolation)
        pos_embed = self.pos_embed
        if x.shape[1] != pos_embed.shape[1]:
            # Interpolate pos_embed to match current resolution
            # pos_embed is (1, N_ref, D) -> (1, D, H_ref, W_ref)
            h_ref = w_ref = int(pos_embed.shape[1] ** 0.5)
            pos_embed = pos_embed.permute(0, 2, 1).view(1, -1, h_ref, w_ref)

            pos_embed = F.interpolate(pos_embed, size=(h_patches, w_patches), mode="bicubic", align_corners=False)

            # (1, D, H', W') -> (1, N, D)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)

        x = x + pos_embed

        # Generate RoPE
        cos, sin = self.rope(h_patches, w_patches, x.device)  # (N, head_dim)
        x = self.encoder(x, cos, sin, attention_mask=attention_mask)
        x = self.norm(x)

        return x
