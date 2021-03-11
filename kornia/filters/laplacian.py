from typing import Tuple

import torch
import torch.nn as nn

import kornia
from kornia.filters.kernels import get_laplacian_kernel2d
from kornia.filters.kernels import normalize_kernel2d


def laplacian(
        input: torch.Tensor,
        kernel_size: int,
        border_type: str = 'reflect',
        normalized: bool = True) -> torch.Tensor:
    r"""Creates an operator that returns a tensor using a Laplacian filter.

    The operator smooths the given tensor with a laplacian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        input (torch.Tensor): the input image tensor with shape :math:`(B, C, H, W)`.
        kernel_size (int): the size of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): if True, L1 norm of the kernel is set to 1.

    Return:
        torch.Tensor: the blurred image with shape :math:`(B, C, H, W)`.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = laplacian(input, 3)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    kernel: torch.Tensor = torch.unsqueeze(
        get_laplacian_kernel2d(kernel_size), dim=0)

    if normalized:
        kernel = normalize_kernel2d(kernel)

    return kornia.filter2D(input, kernel, border_type)


class Laplacian(nn.Module):
    r"""Creates an operator that returns a tensor using a Laplacian filter.

    The operator smooths the given tensor with a laplacian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        kernel_size (int): the size of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): if True, L1 norm of the kernel is set to 1.

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

    def __init__(self,
                 kernel_size: int, border_type: str = 'reflect',
                 normalized: bool = True) -> None:
        super(Laplacian, self).__init__()
        self.kernel_size: int = kernel_size
        self.border_type: str = border_type
        self.normalized: bool = normalized

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(kernel_size=' + str(self.kernel_size) + ', ' +\
            'normalized=' + str(self.normalized) + ', ' + \
            'border_type=' + self.border_type + ')'

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return laplacian(input, self.kernel_size, self.border_type, self.normalized)
