from typing import Tuple

import torch
import torch.nn as nn

import kornia


def laplacian_1d(window_size) -> torch.Tensor:
    r"""One could also use the Laplacian of Gaussian formula
        to design the filter.
    """

    filter_1d = torch.ones(window_size)
    filter_1d[window_size // 2] = 1 - window_size
    laplacian_1d: torch.Tensor = filter_1d
    return laplacian_1d


def get_laplacian_kernel(kernel_size: int) -> torch.Tensor:
    r"""Function that returns the coefficients of a 1D Laplacian filter.

    Args:
        kernel_size (int): filter size. It should be odd and positive.

    Returns:
        Tensor (float): 1D tensor with laplacian filter coefficients.

    Shape:
        - Output: math:`(\text{kernel_size})`

    Examples::
        >>> kornia.image.get_laplacian_kernel(3)
        tensor([ 1., -2.,  1.])

        >>> kornia.image.get_laplacian_kernel(5)
        tensor([ 1.,  1., -4.,  1.,  1.])

    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(kernel_size))
    window_1d: torch.Tensor = laplacian_1d(kernel_size)
    return window_1d


def get_laplacian_kernel2d(kernel_size: int) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (int): filter size should be odd.

    Returns:
        Tensor: 2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])

        >>> kornia.image.get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])

    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(kernel_size))

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d


class Laplacian(nn.Module):
    r"""Creates an operator that returns a tensor using a Laplacian filter.

    The operator smooths the given tensor with a laplacian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        kernel_size (int): the size of the kernel.
        borde_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        Tensor: the tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> laplace = kornia.filters.Laplacian(5)
        >>> output = laplace(input)  # 2x4x5x5
    """

    def __init__(self,
                 kernel_size: int, border_type: str = 'reflect') -> None:
        super(Laplacian, self).__init__()
        self.kernel_size: int = kernel_size
        self.border_type: str = border_type
        self.kernel: torch.Tensor = torch.unsqueeze(
            get_laplacian_kernel2d(kernel_size), dim=0)

    def forward(self, input: torch.Tensor):  # type: ignore
        return kornia.filter2D(input, self.kernel, self.border_type)


######################
# functional interface
######################


def laplacian(
        input: torch.Tensor,
        kernel_size: int, border_type: str = 'reflect') -> torch.Tensor:
    r"""Function that returns a tensor using a Laplacian filter.

    See :class:`~kornia.filters.Laplacian` for details.
    """
    return Laplacian(kernel_size, border_type)(input)
