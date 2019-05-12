from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.functional import conv2d


def _get_box_filter(kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Utility function that returns a box filter."""
    kx: float = float(kernel_size[0])
    ky: float = float(kernel_size[1])
    scale: torch.Tensor = torch.tensor(1.) / torch.tensor([kx * ky])
    tmp_kernel: torch.Tensor = torch.ones(1, 1, kernel_size[0], kernel_size[1])
    return scale.to(tmp_kernel.dtype) * tmp_kernel


def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: Tuple[int, ...] = tuple([(k - 1) // 2 for k in kernel_size])
    return computed[0], computed[1]


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

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = tgm.filters.BoxBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super(BoxBlur, self).__init__()
        self.kernel: torch.Tensor = _get_box_filter(kernel_size)
        self.padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    def forward(self, input: torch.Tensor):  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        return conv2d(input, kernel, padding=self.padding, stride=1, groups=c)


# functiona api


def box_blur(input: torch.Tensor, kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Blurs an image using the box filter.

    See :class:`~torchgeometry.filters.BoxBlur` for details.
    """
    return BoxBlur(kernel_size)(input)
