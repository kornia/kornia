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

"""Normalization layers for transformer models."""

import torch
from torch import nn

from kornia.core import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA-style).

    RMSNorm is a simplification of LayerNorm that removes the mean-centering
    operation, making it more efficient while maintaining similar performance.
    Used in models like LLaMA and other modern transformers.

    Note: This uses the standard parameterization where output = weight * normalized(x)
    with weight initialized to ones. For Gemma-style (weight initialized to zeros,
    output = (1 + weight) * normalized(x)), use GemmaRMSNorm.

    Reference: https://arxiv.org/abs/1910.07467

    Args:
        hidden_size: The dimension of the input features.
        eps: Small constant for numerical stability.

    Example:
        >>> norm = RMSNorm(768)
        >>> x = torch.randn(2, 10, 768)
        >>> output = norm(x)
        >>> output.shape
        torch.Size([2, 10, 768])

    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def _norm(self, x: Tensor) -> Tensor:
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Normalized tensor of the same shape.

        """
        output = self._norm(x.float())
        return (self.weight * output).type_as(x)


class GemmaRMSNorm(nn.Module):
    """Gemma-style Root Mean Square Layer Normalization.

    Identical to RMSNorm but uses the Gemma parameterization where:
    - Weight is initialized to zeros
    - Output = (1 + weight) * normalized(x)

    This is functionally equivalent to RMSNorm at initialization but allows
    direct weight loading from HuggingFace Gemma models.

    Args:
        hidden_size: The dimension of the input features.
        eps: Small constant for numerical stability.

    Example:
        >>> norm = GemmaRMSNorm(768)
        >>> x = torch.randn(2, 10, 768)
        >>> output = norm(x)
        >>> output.shape
        torch.Size([2, 10, 768])

    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def _norm(self, x: Tensor) -> Tensor:
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """Apply Gemma-style RMS normalization.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Normalized tensor of the same shape.

        """
        output = self._norm(x.float())
        # Gemma uses (1 + weight) instead of just weight
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class LayerNorm(nn.Module):
    """Layer Normalization with optional bias.

    Standard layer normalization that normalizes across the last dimension.
    Can optionally exclude the bias term for models that don't use it.

    Args:
        hidden_size: The dimension of the input features.
        eps: Small constant for numerical stability.
        bias: Whether to include a learnable bias term.

    Example:
        >>> norm = LayerNorm(768)
        >>> x = torch.randn(2, 10, 768)
        >>> output = norm(x)
        >>> output.shape
        torch.Size([2, 10, 768])

    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Normalized tensor of the same shape.

        """
        input_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute mean and variance
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)

        x = x.to(input_dtype)

        if self.bias is not None:
            return self.weight * x + self.bias
        return self.weight * x
