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

"""Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention".

Modified from: https://github.com/idiap/fast-
transformers/blob/master/fast_transformers/attention/linear_attention.py.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Dropout


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """Apply elu activation."""
    return torch.nn.functional.elu(x) + 1


def _head_blocks(
    heads: int, dim: int, dim_v: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masks of the per-head blocks: ``(H D, H V)`` block diagonal and the ``(H D, H)`` head of each channel."""
    head_d = torch.arange(heads * dim, device=device) // dim
    head_v = torch.arange(heads * dim_v, device=device) // dim_v
    block = (head_d[:, None] == head_v[None, :]).to(dtype)
    head_of = (head_d[:, None] == torch.arange(heads, device=device)[None, :]).to(dtype)
    return block, head_of


class LinearAttention(nn.Module):
    """Implement the Linear Attention mechanism for efficient transformer processing.

    This module uses the ELU-based kernel map to approximate softmax attention.

    Args:
        eps: A small value to avoid division by zero. Default: 1e-6.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multi-Head linear attention proposed in "Transformers are RNNs".

        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]

        Returns:
            queried_values: (N, L, H, D)

        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        n, length, heads, dim = Q.shape
        dim_v = values.size(3)
        if v_length > heads * dim:
            # Long sequences (LoFTR's coarse level: ~10^4 tokens, 8 x 32 channels). The per-head reductions over the
            # sequence are the diagonal blocks of dense (H D) x (H D) products, which run as single GEMMs instead of
            # an einsum's batched GEMMs with tiny outputs; the off-diagonal blocks are zeroed, not read.
            block, head_of = _head_blocks(heads, dim, dim_v, Q.device, Q.dtype)
            Q = Q.reshape(n, length, heads * dim)
            K = K.reshape(n, v_length, heads * dim)
            KV = (K.transpose(1, 2) @ values.reshape(n, v_length, heads * dim_v)) * block  # (N, H D, H V)
            denominator = Q @ (K.sum(dim=1)[:, :, None] * head_of) + self.eps  # (N, L, H)
        else:
            KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
            denominator = torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps
        if q_mask is not None:
            # A padded query has Q = 0, so it gives 0 whatever its normaliser. Its 1 / eps overflows float16, and
            # 0 * inf would be nan in the output and in the gradient, so it divides by 1 instead.
            denominator = torch.where(q_mask[:, :, None] > 0, denominator, torch.ones_like(denominator))
        Z = 1 / denominator
        if v_length > heads * dim:
            queried_values = (Q @ KV).view(n, length, heads, dim_v) * (Z[..., None] * v_length)
        else:
            # two operands: a three-operand einsum searches for a contraction path on every call
            queried_values = torch.einsum("nlhd,nhdv->nlhv", Q, KV) * (Z[..., None] * v_length)

        return queried_values.contiguous()


class FullAttention(nn.Module):
    """Implement the standard Multi-Head Softmax Attention.

    Args:
        use_dropout: Whether to apply dropout to the attention weights. Default: False.
        attention_dropout: The dropout probability. Default: 0.1.
    """

    def __init__(self, use_dropout: bool = False, attention_dropout: float = 0.1) -> None:
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multi-head scaled dot-product attention, a.k.a full attention.

        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]

        Returns:
            queried_values: (N, L, H, D)

        """
        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None and q_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float("-inf"))

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.size(3) ** 0.5  # sqrt(D)
        A = F.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()
