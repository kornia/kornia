from typing import Tuple

import torch
import torch.nn as nn

import kornia
from kornia.filters.kernels import get_box_kernel2d
from kornia.filters.kernels import normalize_kernel2d


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
        >>> blur = kornia.filters.BoxBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 border_type: str = 'reflect',
                 normalized: bool = True) -> None:
        super(BoxBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.border_type: str = border_type
        self.kernel: torch.Tensor = get_box_kernel2d(kernel_size)
        self.normalized: bool = normalized
        if self.normalized:
            self.kernel = normalize_kernel2d(self.kernel)

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(kernel_size=' + str(self.kernel_size) + ', ' +\
            'normalized=' + str(self.normalized) + ', ' + \
            'border_type=' + self.border_type + ')'

    def forward(self, input: torch.Tensor):  # type: ignore
        return kornia.filter2D(input, self.kernel, self.border_type)


# functiona api


def box_blur(input: torch.Tensor,
             kernel_size: Tuple[int, int],
             border_type: str = 'reflect',
             normalized: bool = True) -> torch.Tensor:
    r"""Blurs an image using the box filter.

    See :class:`~kornia.filters.BoxBlur` for details.
    """
    return BoxBlur(kernel_size, border_type, normalized)(input)
