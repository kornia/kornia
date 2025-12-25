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

"""MLP layers for transformer models."""

import torch.nn.functional as F
from torch import nn

from kornia.core import Tensor


class GeLUMLP(nn.Module):
    """Standard MLP with GeLU activation.

    Two-layer feedforward network commonly used in transformer encoders
    like ViT and SigLIP.

    Args:
        hidden_size: Input and output dimension.
        intermediate_size: Hidden dimension of the MLP.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.

    Example:
        >>> mlp = GeLUMLP(768, 3072)
        >>> x = torch.randn(2, 10, 768)
        >>> output = mlp(x)
        >>> output.shape
        torch.Size([2, 10, 768])

    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of shape (..., hidden_size).

        """
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SwiGLU(nn.Module):
    """SwiGLU feedforward network.

    SwiGLU is a gated linear unit variant that uses the Swish activation.
    Used in modern language models like Gemma and LLaMA.

    The computation is: output = down(swish(gate(x)) * up(x))

    Reference: https://arxiv.org/abs/2002.05202

    Args:
        hidden_size: Input and output dimension.
        intermediate_size: Hidden dimension of the MLP.
        bias: Whether to use bias in linear layers.

    Example:
        >>> mlp = SwiGLU(2048, 16384)
        >>> x = torch.randn(2, 10, 2048)
        >>> output = mlp(x)
        >>> output.shape
        torch.Size([2, 10, 2048])

    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through SwiGLU.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of shape (..., hidden_size).

        """
        # SwiGLU: swish(gate(x)) * up(x)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class GeGLU(nn.Module):
    """GeGLU feedforward network.

    GeGLU is a gated linear unit variant that uses the GeLU activation.
    Similar to SwiGLU but uses GeLU instead of Swish.

    Reference: https://arxiv.org/abs/2002.05202

    Args:
        hidden_size: Input and output dimension.
        intermediate_size: Hidden dimension of the MLP.
        bias: Whether to use bias in linear layers.

    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through GeGLU.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of shape (..., hidden_size).

        """
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        up = self.up_proj(x)
        return self.down_proj(gate * up)
