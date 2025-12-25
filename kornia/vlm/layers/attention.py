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

"""Attention mechanisms for transformer models."""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import Tensor


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    RoPE encodes position information by rotating query and key vectors,
    enabling the model to capture relative positions naturally.

    Reference: https://arxiv.org/abs/2104.09864

    Args:
        dim: Dimension of the embedding (typically head_dim).
        max_position_embeddings: Maximum sequence length.
        base: Base for computing frequencies.

    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        """Precompute cos and sin values for efficiency."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Concatenate to get [cos, cos, sin, sin] pattern
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Get cos and sin embeddings for given positions.

        Args:
            x: Input tensor (used for dtype and device).
            position_ids: Position indices of shape (batch_size, seq_len).

        Returns:
            Tuple of (cos, sin) tensors for the given positions.

        """
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        cos = self.cos_cached[position_ids].to(x.dtype)
        sin = self.sin_cached[position_ids].to(x.dtype)
        return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input.

    Args:
        x: Input tensor of shape (..., dim).

    Returns:
        Rotated tensor where first and second halves are swapped with negation.

    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim).
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim).
        cos: Cosine embeddings of shape (batch, seq_len, head_dim).
        sin: Sine embeddings of shape (batch, seq_len, head_dim).

    Returns:
        Tuple of (rotated_q, rotated_k).

    """
    # Expand cos/sin for broadcasting with num_heads dimension
    cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
    sin = sin.unsqueeze(1)  # (batch, 1, seq_len, head_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional RoPE and Grouped Query Attention.

    Supports standard multi-head attention, grouped query attention (GQA),
    and multi-query attention (MQA) through the num_key_value_heads parameter.

    Args:
        hidden_size: Total hidden dimension.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key-value heads (for GQA/MQA).
            Use same as num_attention_heads for standard MHA.
        head_dim: Dimension of each attention head.
        dropout: Dropout probability for attention weights.
        bias: Whether to use bias in projections.
        use_rotary: Whether to use rotary position embeddings.
        max_position_embeddings: Maximum sequence length for RoPE.
        rope_theta: Base for RoPE frequencies.

    Example:
        >>> attn = MultiHeadAttention(
        ...     hidden_size=2048,
        ...     num_attention_heads=8,
        ...     num_key_value_heads=1,  # GQA
        ...     head_dim=256,
        ... )
        >>> x = torch.randn(2, 10, 2048)
        >>> output, weights = attn(x, output_attentions=True)

    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_rotary: bool = True,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.dropout = dropout
        self.use_rotary = use_rotary

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=bias)

        # Rotary embeddings
        if use_rotary:
            self.rotary_emb = RotaryEmbedding(
                head_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
            )

        # KV cache for generation
        self.k_cache: Optional[Tensor] = None
        self.v_cache: Optional[Tensor] = None

    def reset_cache(self) -> None:
        """Clear the key-value cache."""
        self.k_cache = None
        self.v_cache = None

    def _repeat_kv(self, hidden_states: Tensor, n_rep: int) -> Tensor:
        """Repeat key/value heads for grouped query attention.

        Args:
            hidden_states: Key or value tensor of shape (B, num_kv_heads, seq_len, head_dim).
            n_rep: Number of times to repeat.

        Returns:
            Expanded tensor of shape (B, num_attention_heads, seq_len, head_dim).

        """
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass of multi-head attention.

        Args:
            hidden_states: Input tensor of shape (B, seq_len, hidden_size).
            attention_mask: Optional mask of shape (B, 1, seq_len, seq_len).
            position_ids: Position indices for RoPE, shape (B, seq_len).
            past_key_value: Cached (key, value) from previous step.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return new key-value cache.

        Returns:
            Tuple of:
                - Output tensor of shape (B, seq_len, hidden_size)
                - Attention weights if output_attentions=True, else None
                - Key-value cache if use_cache=True, else None

        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: (B, seq_len, num_heads * head_dim) -> (B, num_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        if self.use_rotary:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            cos, sin = self.rotary_emb(query_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        new_cache = (key_states, value_states) if use_cache else None

        # Repeat K, V for grouped query attention
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back: (B, num_heads, seq_len, head_dim) -> (B, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_attention_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, new_cache
