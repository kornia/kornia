from typing import Tuple

import torch
import torch.nn as nn

from kornia.filters.kernels import get_motion_kernel2d
from kornia.filters.filter import filter2D


class MotionBlur(nn.Module):
    r"""Blurs a tensor using the motion filter.

    Args:
        ksize (int): motion kernel width and height. It should be odd and positive.
        angle (float): angle of the motion blur in degrees (anti-clockwise rotation).
        direction (float): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
        border_type (str): the padding mode to be applied before convolving.
            The expected modes are: ``'constant'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::
        >>> input = torch.rand(2, 4, 5, 7)
        >>> motion_blur = kornia.filters.MotionBlur(3, 35., 0.5)
        >>> output = motion_blur(input)  # 2x4x5x7
    """

    def __init__(
            self, ksize: int, angle: float, direction: float, border_type: str = 'constant'
    ) -> None:
        super(MotionBlur, self).__init__()
        self.ksize = ksize
        self.angle: float = angle
        self.direction: float = direction
        self.border_type: str = border_type

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} (ksize={self.ksize}, ' \
               f'angle={self.angle}, direction={self.direction})'

    def forward(self, x: torch.Tensor):  # type: ignore
        return filter2D(x, self.kernel, self.angle, self.direction, self.border_type)


def motion_blur(
    input: torch.Tensor,
    ksize: int,
    angle: float,
    direction: float,
    border_type: str = 'constant'
) -> torch.Tensor:
    r"""
    Function that blurs a tensor using the motion filter.

    See :class:`~kornia.filters.MotionBlur` for details.
    """
    assert border_type in ["constant", "reflect", "replicate", "circular"]
    kernel: torch.Tensor = torch.unsqueeze(
        get_motion_kernel2d(ksize, angle, direction), dim=0)
    return filter2D(input, kernel, border_type)
