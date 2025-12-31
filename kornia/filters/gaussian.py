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

from typing import Any

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.utils import deprecated

from .filter import filter2d, filter2d_separable
from .kernels import _check_kernel_size, _unpack_2d_ks, get_gaussian_kernel1d, get_gaussian_kernel2d


def gaussian_blur2d(
    input: torch.Tensor,
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | torch.Tensor,
    border_type: str = "reflect",
    separable: bool = True,
) -> torch.Tensor:
    r"""Create an operator that blurs a torch.tensor using a Gaussian filter.

    .. image:: _static/img/gaussian_blur2d.png

    The operator smooths the given torch.tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        input: the input torch.tensor with shape :math:`(B,C,H,W)`.
        kernel_size: the size of the kernel. Can be an integer or tuple of two integers (height, width).
        sigma: the standard deviation of the kernel. Can be a tuple of two floats or a torch.tensor
            with shape :math:`(B, 2)`.
          Values must be positive.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: run as composition of two 1d-convolutions. Default: ``True``.

    Returns:
        the blurred torch.tensor with shape :math:`(B, C, H, W)`.

    Raises:
        RuntimeError: if input is not a 4D torch.tensor.
        RuntimeError: if sigma values are not positive.
        RuntimeError: if kernel_size is not a positive odd integer.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/gaussian_blur.html>`__.

    Examples:
        >>> import torch
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = gaussian_blur2d(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])

        >>> # Single kernel size applies to both dimensions
        >>> output = gaussian_blur2d(input, 3, (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])

        >>> # Using batched sigma (different sigma per batch element)
        >>> sigma_batch = torch.tensor([[1.5, 1.5], [2.0, 2.0]])
        >>> output = gaussian_blur2d(input[:2], (3, 3), sigma_batch)
        >>> output.shape
        torch.Size([2, 4, 5, 5])

        >>> # Using torch.tensor sigma
        >>> output = gaussian_blur2d(input, (3, 3), torch.tensor([[1.5, 1.5]]))
        >>> output.shape
        torch.Size([2, 4, 5, 5])

    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    _check_kernel_size(kernel_size, min_value=0)

    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], device=input.device, dtype=input.dtype)
    else:
        KORNIA_CHECK_IS_TENSOR(sigma)
        sigma = sigma.to(device=input.device, dtype=input.dtype)

    # Validate sigma values are positive
    KORNIA_CHECK_SHAPE(sigma, ["B", "2"])
    KORNIA_CHECK(bool((sigma > 0).all()), f"sigma must be positive, got {sigma}")

    if separable:
        ky, kx = _unpack_2d_ks(kernel_size)
        bs = sigma.shape[0]
        kernel_x = get_gaussian_kernel1d(kx, sigma[:, 1].view(bs, 1))
        kernel_y = get_gaussian_kernel1d(ky, sigma[:, 0].view(bs, 1))
        out = filter2d_separable(input, kernel_x, kernel_y, border_type)
    else:
        kernel = get_gaussian_kernel2d(kernel_size, sigma)
        out = filter2d(input, kernel, border_type)

    return out


class GaussianBlur2d(nn.Module):
    r"""Create an operator that blurs a torch.tensor using a Gaussian filter.

    The operator smooths the given torch.tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: run as composition of two 1d-convolutions.

    Returns:
        the blurred torch.tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = GaussianBlur2d((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
        >>> output.shape
        torch.Size([2, 4, 5, 5])

    """

    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        sigma: tuple[float, float] | torch.Tensor,
        border_type: str = "reflect",
        separable: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.border_type = border_type
        self.separable = separable

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"sigma={self.sigma}, "
            f"border_type={self.border_type}, "
            f"separable={self.separable})"
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return gaussian_blur2d(input, self.kernel_size, self.sigma, self.border_type, self.separable)


@deprecated(replace_with="gaussian_blur2d", version="6.9.10")
def gaussian_blur2d_t(*args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: D103
    return gaussian_blur2d(*args, **kwargs)
