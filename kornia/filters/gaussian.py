from typing import Tuple

import torch
import torch.nn as nn

import kornia
from kornia.filters.kernels import get_gaussian_kernel2d


def gaussian_blur2d(
        input: torch.Tensor,
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        border_type: str = 'reflect') -> torch.Tensor:
    r"""Creates an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        input (torch.Tensor): the input tensor with shape :math:`(B,C,H,W)`.
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        torch.Tensor: the blurred tensor with shape :math:`(B, C, H, W)`.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = gaussian_blur2d(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    kernel: torch.Tensor = torch.unsqueeze(
        get_gaussian_kernel2d(kernel_size, sigma), dim=0)

    return kornia.filter2D(input, kernel, border_type)


class GaussianBlur2d(nn.Module):
    r"""Creates an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        Tensor: the blurred tensor.

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

    def __init__(self, kernel_size: Tuple[int, int],
                 sigma: Tuple[float, float],
                 border_type: str = 'reflect') -> None:
        super(GaussianBlur2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self.border_type = border_type

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(kernel_size=' + str(self.kernel_size) + ', ' +\
            'sigma=' + str(self.sigma) + ', ' +\
            'border_type=' + self.border_type + ')'

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return gaussian_blur2d(input, self.kernel_size, self.sigma, self.border_type)
