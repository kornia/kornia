from typing import Tuple

import torch
import torch.nn as nn

import kornia
from kornia.filters.kernels import get_gaussian_kernel2d


class GaussianBlur2d(nn.Module):
    r"""Creates an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        borde_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = kornia.filters.GaussianBlur((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 sigma: Tuple[float, float],
                 border_type: str = 'reflect') -> None:
        super(GaussianBlur2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self.kernel: torch.Tensor = torch.unsqueeze(
            get_gaussian_kernel2d(kernel_size, sigma), dim=0)

        assert border_type in ["constant", "reflect", "replicate", "circular"]
        self.border_type = border_type

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(kernel_size=' + str(self.kernel_size) + ', ' +\
            'sigma=' + str(self.sigma) + ', ' +\
            'border_type=' + self.border_type + ')'

    def forward(self, x: torch.Tensor):  # type: ignore
        return kornia.filter2D(x, self.kernel, self.border_type)


######################
# functional interface
######################


def gaussian_blur2d(
        input: torch.Tensor,
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        border_type: str = 'reflect') -> torch.Tensor:
    r"""Function that blurs a tensor using a Gaussian filter.

    See :class:`~kornia.filters.GaussianBlur` for details.
    """
    return GaussianBlur2d(kernel_size, sigma, border_type)(input)
