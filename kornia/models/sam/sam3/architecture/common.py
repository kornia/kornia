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
from torch import nn


class LayerNorm(nn.Module):
    """Layer normalization with optional absolute position embedding.

    This implementation allows for layer normalization with optional positional embedding support.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """Initialize LayerNorm.

        Args:
            num_channels: Number of channels (features) to normalize.
            eps: Small value for numerical stability in normalization.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor of shape (..., C) where C is the number of channels.

        Returns:
            Normalized tensor of the same shape as input.
        """
        u = x.mean(dim=-1, keepdim=True)
        s = (x - u).pow(2).mean(dim=-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


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
            dim: Dimension of the input features.
            heads: Number of attention heads.
            dim_head: Dimension of each attention head.
            dropout: Dropout probability.
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
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

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.to_out(out)


def get_activation_fn(activation: str) -> nn.Module:
    """Get activation function by name.

    Args:
        activation: Name of the activation function ('relu', 'gelu', 'gelu_approx', etc.)

    Returns:
        The requested activation module.

    Raises:
        RuntimeError: If the activation name is not supported.
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "gelu_approx":
        return nn.GELU(approximate="tanh")
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise RuntimeError(f"Activation '{activation}' not supported")


__all__ = ["Attention", "LayerNorm", "MLPBlock", "get_activation_fn"]
