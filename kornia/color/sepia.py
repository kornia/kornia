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
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK_SHAPE


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
    # Safety Checks
    KORNIA_CHECK_SHAPE(input, ["*", "3", "H", "W"])

    image_compute = input if input.is_floating_point() else input.float()
    input_shape = image_compute.shape

    # Standard Sepia Matrix
    kernel = torch.tensor(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ],
        device=image_compute.device,
        dtype=image_compute.dtype,
    )

    # Empirical benchmarks show that einsum is faster on CPU for this specific pattern,
    # while conv2d offers significant speedups on GPU/CUDA.
    # We branch to ensure optimal performance on both devices.
    # BRANCH 1: CPU (Einsum)
    if input.device.type == "cpu":
        out = torch.einsum("...chw,oc->...ohw", image_compute, kernel)
        out = out.contiguous()
    # BRANCH 2: GPU/Accelerators (Conv2d)
    else:
        # Reshape for conv2d: (B*..., C, H, W)
        input_flat = image_compute.reshape(-1, 3, input_shape[-2], input_shape[-1])

        # Reshape kernel: (3, 3) -> (3, 3, 1, 1)
        weight = kernel.view(3, 3, 1, 1)

        out_flat = F.conv2d(input_flat, weight)

        # Unflatten back to original shape
        out = out_flat.reshape(input_shape)

    if rescale:
        max_values = out.amax(dim=-1).amax(dim=-1)
        out = out / (max_values[..., None, None] + eps)

    return out


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
