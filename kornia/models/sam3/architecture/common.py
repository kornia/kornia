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

"""Common components for SAM-3 architecture.

This module provides shared building blocks for SAM-3 including:
- Normalization layers
- Activation functions
- Attention primitives
- Helper utilities
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLPBlock(nn.Module):
    """Multi-layer Perceptron block.

    A simple feedforward network with two linear layers and GELU activation.
    """

    def __init__(self, embedding_dim: int, mlp_dim: int) -> None:
        """Initialize MLPBlock.

        Args:
            embedding_dim: Dimension of input and output features.
            mlp_dim: Dimension of the hidden layer.
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor of the same shape as input.
        """
        return self.lin2(self.act(self.lin1(x)))


class Attention(nn.Module):
    """Multi-head attention block.

    Standard multi-head self-attention implementation following the Transformer architecture.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        """Initialize Attention.

        Args:
            dim: Dimension of the input features. Should be divisible by heads * dim_head if project_out=False.
            heads: Number of attention heads.
            dim_head: Dimension of each attention head.
            dropout: Dropout probability.
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention.

        Args:
            x: Input tensor of shape (B, N, D) where B is batch size, N is sequence length, D is feature dimension.

        Returns:
            Output tensor of the same shape as input.
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = tuple(t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2) for t in qkv)

        dropout_p = self.dropout.p if self.training and self.dropout.p > 0.0 else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.to_out(out)


__all__ = ["Attention", "MLPBlock"]
