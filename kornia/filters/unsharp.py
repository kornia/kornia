from typing import Tuple

import torch
import torch.nn as nn

import kornia
from kornia.filters import gaussian_blur2d


def unsharp_mask(
        input: torch.Tensor,
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        border_type: str = 'reflect') -> torch.Tensor:
    r"""Creates an operator that blurs a tensor using the existing Gaussian filter available with the Kornia library.

    Arguments:
        input (torch.Tensor): the input tensor with shape :math:`(B,C,H,W)`.
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        torch.Tensor: the blurred tensor with shape :math:`(B,C,H,W)`.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = unsharp_mask(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    data_blur: torch.Tensor = gaussian_blur2d(input, kernel_size, sigma)
    data_sharpened: torch.Tensor = input + (input - data_blur)
    return data_sharpened


class UnsharpMask(nn.Module):
    r"""Creates an operator that sharpens image using the existing Gaussian filter available with the Kornia library..

    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        Tensor: the sharpened tensor with shape :math:`(B,C,H,W)`.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:

        >>> input = torch.rand(2, 4, 5, 5)
        >>> sharpen = UnsharpMask((3, 3), (1.5, 1.5))
        >>> output = sharpen(input)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 sigma: Tuple[float, float],
                 border_type: str = 'reflect') -> None:
        super(UnsharpMask, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self.border_type = border_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return unsharp_mask(input, self.kernel_size, self.sigma, self.border_type)
