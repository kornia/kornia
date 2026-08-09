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

import torch
from torch import nn

from .filter import filter2d
from .kernels import get_laplacian_kernel2d, normalize_kernel2d


def laplacian(
    input: torch.Tensor, kernel_size: tuple[int, int] | int, border_type: str = "reflect", normalized: bool = True
) -> torch.Tensor:
    r"""Create an operator that returns a tensor using a Laplacian filter.

    .. image:: _static/img/laplacian.png

    The operator smooths the given tensor with a laplacian kernel by convolving
    it to each channel. It supports batched operation.

    Args:
        input: the input image tensor with shape :math:`(B, C, H, W)`.
        kernel_size: the size of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: if True, L1 norm of the kernel is set to 1.

    Return:
        the blurred image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/filtering_edges.html>`__.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = laplacian(input, 3)
        >>> output.shape
        torch.Size([2, 4, 5, 5])

    """
    kernel = get_laplacian_kernel2d(kernel_size, device=input.device, dtype=input.dtype)[None, ...]

    if normalized:
        kernel = normalize_kernel2d(kernel)

    return filter2d(input, kernel, border_type)


class Laplacian(nn.Module):
    r"""Create an operator that returns a tensor using a Laplacian filter.

    The operator smooths the given tensor with a laplacian kernel by convolving
    it to each channel. It supports batched operation.

    Args:
        kernel_size: the size of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: if True, L1 norm of the kernel is set to 1.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> laplace = Laplacian(5)
        >>> output = laplace(input)
        >>> output.shape
        torch.Size([2, 4, 5, 5])

    """

    def __init__(
        self, kernel_size: tuple[int, int] | int, border_type: str = "reflect", normalized: bool = True
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.border_type: str = border_type
        self.normalized: bool = normalized

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"normalized={self.normalized}, "
            f"border_type={self.border_type})"
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the second-order Laplacian response of an image tensor.

        The Laplacian filter measures rapid local intensity changes by
        combining second derivatives along the spatial axes. It is commonly
        used for edge detection, focus measures, and highlighting fine image
        detail.

        Args:
            input: Image tensor with shape :math:`(B, C, H, W)`, where
                :math:`B` is the batch size, :math:`C` is the number of
                channels, :math:`H` is the height, and :math:`W` is the width.

        Returns:
            Tensor with shape :math:`(B, C, H, W)` containing the Laplacian
            response for each batch item and channel. Positive and negative
            values represent opposite directions of local curvature.
        """
        return laplacian(input, self.kernel_size, self.border_type, self.normalized)
