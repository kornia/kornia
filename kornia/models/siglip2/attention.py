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

"""Attention modules for SigLip2."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK

__all__ = ["SigLip2Attention"]


class SigLip2Attention(nn.Module):
    """Multi-head self-attention mechanism for SigLip2.

    This module implements the standard multi-head self-attention used in
    transformer architectures, with support for attention masks.

    Args:
        hidden_size: Hidden dimension size.
        num_heads: Number of attention heads.
        dropout: Dropout probability for attention weights.
        head_dim: Dimension of each attention head. If None, computed as hidden_size // num_heads.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_p: float = 0.0,
        head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        if head_dim is None:
            head_dim = hidden_size // num_heads
        self.head_dim = head_dim

        KORNIA_CHECK(
            hidden_size % num_heads == 0,
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})",
        )

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Separate Q/K/V projections (matching HF structure)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of attention.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask, where ``1``/``True`` means attend and ``0``/``False``
                means mask out. Can be:

                - (batch_size, seq_len): a padding mask, broadcast over the key axis only, matching
                  ``SiglipTextTransformer.create_bidirectional_mask`` in Hugging Face ``transformers``.
                  A padded query row is therefore *not* masked and its output is finite but meaningless;
                  callers are expected to discard those positions.
                - (batch_size, seq_len, seq_len): a per-query mask, broadcast over the head axis.
                - (batch_size, 1, seq_len, seq_len): already in the layout
                  ``torch.nn.functional.scaled_dot_product_attention`` expects.

                A query row whose mask is entirely ``0`` follows the behavior of
                ``scaled_dot_product_attention`` for a fully masked row: ``0`` on PyTorch 2.5 and later,
                ``nan`` before it.

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # compute Q, K, V separately
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # reshape to (batch_size, seq_len, num_heads, head_dim)
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # transpose to (batch_size, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Convert attention mask to format expected by scaled_dot_product_attention
        attn_mask = None
        if attention_mask is not None:
            # Handle different mask formats
            if attention_mask.dim() == 2:
                # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                # Mask keys only so padded query positions do not create fully masked
                # rows, which produce NaNs in older SDPA implementations.
                # Boolean SDPA masks use True = attend and False = mask out.
                attn_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                attn_mask = attention_mask.bool().unsqueeze(1)
            elif attention_mask.dim() == 4:
                # Already in the SDPA layout (batch_size, 1, seq_len, seq_len); True = attend
                attn_mask = attention_mask.bool()
            else:
                raise ValueError(f"Unsupported attention_mask dimension: {attention_mask.dim()}")

        dropout_p = self.dropout.p if self.training and self.dropout.p > 0.0 else 0.0
        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attention_output)
