from typing import Tuple

import torch
import torch.nn as nn

import kornia
from kornia.filters.kernels import (
    get_box_kernel2d, normalize_kernel2d
)


def box_blur(input: torch.Tensor,
             kernel_size: Tuple[int, int],
             border_type: str = 'reflect',
             normalized: bool = True) -> torch.Tensor:
    r"""Blurs an image using the box filter.

    The function smooths an image using the kernel:

    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Args:
        image (torch.Tensor): the image to blur with shape :math:`(B,C,H,W)`.
        kernel_size (Tuple[int, int]): the blurring kernel size.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): if True, L1 norm of the kernel is set to 1.

    Returns:
        torch.Tensor: the blurred tensor with shape :math:`(B,C,H,W)`.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> output = box_blur(input, (3, 3))  # 2x4x5x7
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    kernel: torch.Tensor = get_box_kernel2d(kernel_size)
    if normalized:
        kernel = normalize_kernel2d(kernel)
    return kornia.filter2D(input, kernel, border_type)


class BoxBlur(nn.Module):
    r"""Blurs an image using the box filter.

    The function smooths an image using the kernel:

    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): if True, L1 norm of the kernel is set to 1.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = BoxBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 border_type: str = 'reflect',
                 normalized: bool = True) -> None:
        super(BoxBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.border_type: str = border_type
        self.normalized: bool = normalized

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(kernel_size=' + str(self.kernel_size) + ', ' +\
            'normalized=' + str(self.normalized) + ', ' + \
            'border_type=' + self.border_type + ')'

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return box_blur(input, self.kernel_size, self.border_type, self.normalized)
