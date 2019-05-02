from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.functional import conv2d


def laplacian_1d(window_size) -> torch.Tensor:
    r"""One could also use the Laplacian of Gaussian formula
        to design the filter.
    """

    filter_1d = torch.ones(window_size)
    filter_1d[window_size // 2] = 1 - window_size
    laplacian_1d: torch.Tensor = filter_1d
    return laplacian_1d


def get_laplacian_kernel(kernel_size: int) -> torch.Tensor:
    r"""Function that returns the coefficients of a 1D Laplacian filter

    Args:
        kernel_size (int): filter size. It should be odd and positive.

    Returns:
        Tensor (float): 1D tensor with laplacian filter coefficients.

    Shape:
        - Output: math:`(ksize, 0)`

    Examples::
        >>> tgm.image.get_laplacian_kernel(3)
        tensor([ 1., -2.,  1.])

        >>> tgm.image.get_laplacian_kernel(5)
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
        ksize (int): filter size should be odd.

    Returns:
        Tensor: 2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(ksize, ksize)`

    Examples::

        >>> tgm.image.get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
        [ 1., -8.,  1.],
        [ 1.,  1.,  1.]])

        >>> tgm.image.get_laplacian_kernel2d(5)
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

    Returns:
        Tensor: the tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> laplace = tgm.image.Laplacian(5)
        >>> output = laplace(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size: int) -> None:
        super(Laplacian, self).__init__()
        self.kernel_size: int = kernel_size
        self._padding: int = self.compute_zero_padding(kernel_size)
        self.kernel: torch.Tensor = self.get_laplacian_kernel(kernel_size)

    @staticmethod
    def get_laplacian_kernel(kernel_size) -> torch.Tensor:
        """Returns a 2D Laplacian kernel array."""
        kernel: torch.Tensor = get_laplacian_kernel2d(kernel_size)
        return kernel

    @staticmethod
    def compute_zero_padding(kernel_size: int):
        """Computes zero padding."""
        computed = (kernel_size - 1) // 2
        return computed

    def forward(self, x: torch.Tensor):  # type: ignore
        if not torch.is_tensor(x):
            raise TypeError("Input x type is not a torch.Tensor. Got {}"
                            .format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(x.shape))
        # prepare kernel
        b, c, h, w = x.shape
        tmp_kernel: torch.Tensor = self.kernel.to(x.device).to(x.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # convolve tensor with gaussian kernel
        # TODO: Filter2D convolves an image with the kernel. A wrapper fucntion
        #  for conv2d something similar to the one
        #       given in OpenCV. Also see separable filters.
        # https://github.com/opencv/opencv/blob/7fb70e170154d064ef12d8fec61c0ae70812ce3d/modules/imgproc/src/deriv.cpp#L822
        return conv2d(x, kernel, padding=self._padding, stride=1, groups=c)


######################
# functional interface
######################


def laplacian(src: torch.Tensor, kernel_size: int) -> torch.Tensor:
    r"""Function that returns a tensor using a Laplacian filter.

    The operator smooths the given tensor with a laplacian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        src (torch.Tensor): the input tensor.
        kernel_size (int): the size of the kernel.

    Returns:
        Tensor: the tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = tgm.image.laplacian(input, (3, 3), (1.5, 1.5))
    """
    return Laplacian(kernel_size)(src)
