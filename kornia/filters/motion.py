from typing import Tuple, Union

import torch
import torch.nn as nn

from kornia.filters.kernels import get_motion_kernel2d, get_motion_kernel3d
from kornia.filters.filter import filter2D, filter3D


class MotionBlur(nn.Module):
    r"""Blur 2D images (4D tensor) using the motion filter.

    Args:
        kernel_size (int): motion kernel width and height. It should be odd and positive.
        angle (float): angle of the motion blur in degrees (anti-clockwise rotation).
        direction (float): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
        border_type (str): the padding mode to be applied before convolving. The expected modes are:
            ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'constant'``.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> motion_blur = MotionBlur(3, 35., 0.5)
        >>> output = motion_blur(input)  # 2x4x5x7
    """

    def __init__(
            self, kernel_size: int, angle: float, direction: float, border_type: str = 'constant'
    ) -> None:
        super(MotionBlur, self).__init__()
        self.kernel_size = kernel_size
        self.angle: float = angle
        self.direction: float = direction
        self.border_type: str = border_type

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} (kernel_size={self.kernel_size}, ' \
               f'angle={self.angle}, direction={self.direction}, border_type={self.border_type})'

    def forward(self, x: torch.Tensor):  # type: ignore
        return motion_blur(x, self.kernel_size, self.angle, self.direction, self.border_type)


class MotionBlur3D(nn.Module):
    r"""Blur 3D volumes (5D tensor) using the motion filter.

    Args:
        kernel_size (int): motion kernel width and height. It should be odd and positive.
        angle (float or tuple): Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
        direction (float): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
        border_type (str): the padding mode to be applied before convolving. The expected modes are:
            ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'constant'``.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, D, H, W)`
        - Output: :math:`(B, C, D, H, W)`

    Examples:
        >>> input = torch.rand(2, 4, 5, 7, 9)
        >>> motion_blur = MotionBlur3D(3, 35., 0.5)
        >>> output = motion_blur(input)  # 2x4x5x7x9
    """

    def __init__(
            self, kernel_size: int, angle: Union[float, Tuple[float, float, float]],
            direction: float, border_type: str = 'constant'
    ) -> None:
        super(MotionBlur3D, self).__init__()
        self.kernel_size = kernel_size
        self.angle: Tuple[float, float, float]
        if isinstance(angle, float):
            self.angle = (angle, angle, angle)
        elif isinstance(angle, (tuple, list)) and len(angle) == 3:
            self.angle = angle
        else:
            raise ValueError(f"Expect angle to be either a float or a tuple of floats. Got {angle}.")
        self.direction: float = direction
        self.border_type: str = border_type

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} (kernel_size={self.kernel_size}, ' \
               f'angle={self.angle}, direction={self.direction}, border_type={self.border_type})'

    def forward(self, x: torch.Tensor):  # type: ignore
        return motion_blur3d(x, self.kernel_size, self.angle, self.direction, self.border_type)


def motion_blur(
    input: torch.Tensor,
    kernel_size: int,
    angle: Union[float, torch.Tensor],
    direction: Union[float, torch.Tensor],
    border_type: str = 'constant'
) -> torch.Tensor:
    r"""Perform motion blur on 2D images (4D tensor).

    Args:
        input (tensor): the input tensor with shape :math:`(B, C, H, W)`.
        kernel_size (int): motion kernel width and height. It should be odd and positive.
        angle (tensor, float): angle of the motion blur in degrees (anti-clockwise rotation).
            If tensor, it must be :math:`(B,)`.
        direction (tensor or float): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
            If tensor, it must be :math:`(B,)`.
        border_type (str): the padding mode to be applied before convolving. The expected modes are:
            ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'constant'``.

    See :class:`~kornia.filters.MotionBlur` for details.

    Example:
        >>> input = torch.randn(1, 3, 80, 90).repeat(2, 1, 1, 1)
        >>> # perform exact motion blur accross the batch
        >>> out_1 = motion_blur(input, 5, 90., 1)
        >>> torch.allclose(out_1[0], out_1[1])
        True
        >>> # perform element-wise motion blur accross the batch
        >>> out_1 = motion_blur(input, 5, torch.tensor([90., 180,]), torch.tensor([1., -1.]))
        >>> torch.allclose(out_1[0], out_1[1])
        False
    """
    assert border_type in ["constant", "reflect", "replicate", "circular"]
    kernel: torch.Tensor = get_motion_kernel2d(kernel_size, angle, direction)
    return filter2D(input, kernel, border_type)


def motion_blur3d(
    input: torch.Tensor,
    kernel_size: int,
    angle: Union[Tuple[float, float, float], torch.Tensor],
    direction: Union[float, torch.Tensor],
    border_type: str = 'constant'
) -> torch.Tensor:
    r"""Perform motion blur on 3D volumes (5D tensor).

    Args:
        input (tensor): the input tensor with shape :math:`(B, C, D, H, W)`.
        kernel_size (int): motion kernel width, height and depth. It should be odd and positive.
        angle (tensor or tuple): Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
            If tensor, it must be :math:`(B, 3)`.
        direction (tensor or float): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
            If tensor, it must be :math:`(B,)`.
        border_type (str): the padding mode to be applied before convolving. The expected modes are:
            ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'constant'``.

    Example:
        >>> input = torch.randn(1, 3, 120, 80, 90).repeat(2, 1, 1, 1, 1)
        >>> # perform exact motion blur accross the batch
        >>> out_1 = motion_blur3d(input, 5, (0., 90., 90.), 1)
        >>> torch.allclose(out_1[0], out_1[1])
        True
        >>> # perform element-wise motion blur accross the batch
        >>> out_1 = motion_blur3d(input, 5, torch.tensor([[0., 90., 90.], [90., 180., 0.]]), torch.tensor([1., -1.]))
        >>> torch.allclose(out_1[0], out_1[1])
        False
    """
    assert border_type in ["constant", "reflect", "replicate", "circular"]
    kernel: torch.Tensor = get_motion_kernel3d(kernel_size, angle, direction)
    return filter3D(input, kernel, border_type)
