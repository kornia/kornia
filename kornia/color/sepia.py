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

import torch
from torch import nn
import torch.nn.functional as F


def sepia_from_rgb(input: torch.Tensor, rescale: bool = True, eps: float = 1e-6) -> torch.Tensor:
    r"""Apply to a torch.Tensor the sepia filter.

    Args:
        input: the input torch.Tensor with shape of :math:`(*, 3, H, W)`.
        rescale: If True, the output torch.Tensor will be rescaled (max values be 1. or 255).
        eps: scalar to enforce numerical stability.

    Returns:
        torch.Tensor: The sepia torch.tensor of same size and numbers of channels
        as the input with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.ones(3, 1, 1)
        >>> sepia_from_rgb(input, rescale=False)
        tensor([[[1.3510]],
        <BLANKLINE>
                [[1.2030]],
        <BLANKLINE>
                [[0.9370]]])
    """
    if len(input.shape) < 3 or input.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {input.shape}")

    # Standard Sepia Matrix
    # Row 0: R, Row 1: G, Row 2: B
    kernel = torch.tensor(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ],
        device=input.device,
        dtype=input.dtype,
    )

    # 1. CPU Strategy: Einsum (Memory Efficient)
    if input.device.type == "cpu":
        sepia_out = torch.einsum("...chw,oc->...ohw", input, kernel)

    # 2. GPU Strategy: Conv2d (Compute Efficient)
    else:
        # conv2d requires 4D input (B, C, H, W).
        
        # We flatten arbitrary batch dims into B, apply conv, then unflatten.
        input_shape = input.shape
        input_flat = input.view(-1, 3, input_shape[-2], input_shape[-1])
        weight = kernel.view(3, 3, 1, 1)
        sepia_out_flat = F.conv2d(input_flat, weight)
        
        # Reshape back to original (*, 3, H, W)
        sepia_out = sepia_out_flat.view(*input_shape)

    if rescale:
        max_values = sepia_out.amax(dim=-1).amax(dim=-1)
        sepia_out = sepia_out / (max_values[..., None, None] + eps)

    return sepia_out


class Sepia(nn.Module):
    r"""nn.Module that apply the sepia filter to tensors.

    Args:
        input: the input torch.Tensor with shape of :math:`(*, C, H, W)`.
        rescale: If True, the output torch.Tensor will be rescaled (max values be 1. or 255).
        eps: scalar to enforce numerical stability.

    Returns:
        torch.Tensor: The sepia torch.tensor of same size and numbers of channels
        as the input with shape :math:`(*, C, H, W)`.

    Example:
        >>>
        >>> input = torch.ones(3, 1, 1)
        >>> Sepia(rescale=False)(input)
        tensor([[[1.3510]],
        <BLANKLINE>
                [[1.2030]],
        <BLANKLINE>
                [[0.9370]]])

    """

    def __init__(self, rescale: bool = True, eps: float = 1e-6) -> None:
        self.rescale = rescale
        self.eps = eps
        super().__init__()

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(rescale={self.rescale}, eps={self.eps})"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return sepia_from_rgb(input, rescale=self.rescale, eps=self.eps)
