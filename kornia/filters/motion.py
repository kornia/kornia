from __future__ import annotations

from kornia.core import Module, Tensor

from .filter import filter2d, filter3d
from .kernels_geometry import get_motion_kernel2d, get_motion_kernel3d


class MotionBlur(Module):
    r"""Blur 2D images (4D tensor) using the motion filter.

    Args:
        kernel_size: motion kernel width and height. It should be odd and positive.
        angle: angle of the motion blur in degrees (anti-clockwise rotation).
        direction: forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
        border_type: the padding mode to be applied before convolving. The expected modes are:
            ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.

    Returns:
        the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> motion_blur = MotionBlur(3, 35., 0.5)
        >>> output = motion_blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size: int, angle: float, direction: float, border_type: str = 'constant') -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.angle: float = angle
        self.direction: float = direction
        self.border_type: str = border_type

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__} (kernel_size={self.kernel_size}, '
            f'angle={self.angle}, direction={self.direction}, border_type={self.border_type})'
        )

    def forward(self, x: Tensor):
        return motion_blur(x, self.kernel_size, self.angle, self.direction, self.border_type)


class MotionBlur3D(Module):
    r"""Blur 3D volumes (5D tensor) using the motion filter.

    Args:
        kernel_size: motion kernel width and height. It should be odd and positive.
        angle: Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
        direction: forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
        border_type: the padding mode to be applied before convolving. The expected modes are:
            ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.

    Returns:
        the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, D, H, W)`
        - Output: :math:`(B, C, D, H, W)`

    Examples:
        >>> input = torch.rand(2, 4, 5, 7, 9)
        >>> motion_blur = MotionBlur3D(3, 35., 0.5)
        >>> output = motion_blur(input)  # 2x4x5x7x9
    """

    def __init__(
        self,
        kernel_size: int,
        angle: float | tuple[float, float, float],
        direction: float,
        border_type: str = 'constant',
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.angle: tuple[float, float, float]
        if isinstance(angle, float):
            self.angle = (angle, angle, angle)
        elif isinstance(angle, (tuple, list)) and len(angle) == 3:
            self.angle = angle
        else:
            raise ValueError(f"Expect angle to be either a float or a tuple of floats. Got {angle}.")
        self.direction: float = direction
        self.border_type: str = border_type

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__} (kernel_size={self.kernel_size}, '
            f'angle={self.angle}, direction={self.direction}, border_type={self.border_type})'
        )

    def forward(self, x: Tensor):
        return motion_blur3d(x, self.kernel_size, self.angle, self.direction, self.border_type)


def motion_blur(
    input: Tensor,
    kernel_size: int,
    angle: float | Tensor,
    direction: float | Tensor,
    border_type: str = 'constant',
    mode: str = 'nearest',
) -> Tensor:
    r"""Perform motion blur on tensor images.

    .. image:: _static/img/motion_blur.png

    Args:
        input: the input tensor with shape :math:`(B, C, H, W)`.
        kernel_size: motion kernel width and height. It should be odd and positive.
        angle (Union[torch.Tensor, float]): angle of the motion blur in degrees (anti-clockwise rotation).
            If tensor, it must be :math:`(B,)`.
        direction : forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
            If tensor, it must be :math:`(B,)`.
        border_type: the padding mode to be applied before convolving. The expected modes are:
            ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'constant'``.
        mode: interpolation mode for rotating the kernel. ``'bilinear'`` or ``'nearest'``.

    Return:
        the blurred image with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.randn(1, 3, 80, 90).repeat(2, 1, 1, 1)
        >>> # perform exact motion blur across the batch
        >>> out_1 = motion_blur(input, 5, 90., 1)
        >>> torch.allclose(out_1[0], out_1[1])
        True
        >>> # perform element-wise motion blur across the batch
        >>> out_1 = motion_blur(input, 5, torch.tensor([90., 180,]), torch.tensor([1., -1.]))
        >>> torch.allclose(out_1[0], out_1[1])
        False
    """
    if border_type not in ["constant", "reflect", "replicate", "circular"]:
        raise AssertionError
    kernel: Tensor = get_motion_kernel2d(kernel_size, angle, direction, mode)
    return filter2d(input, kernel, border_type)


def motion_blur3d(
    input: Tensor,
    kernel_size: int,
    angle: tuple[float, float, float] | Tensor,
    direction: float | Tensor,
    border_type: str = 'constant',
    mode: str = 'nearest',
) -> Tensor:
    r"""Perform motion blur on 3D volumes (5D tensor).

    Args:
        input: the input tensor with shape :math:`(B, C, D, H, W)`.
        kernel_size: motion kernel width, height and depth. It should be odd and positive.
        angle: Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
            If tensor, it must be :math:`(B, 3)`.
        direction: forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
            If tensor, it must be :math:`(B,)`.
        border_type: the padding mode to be applied before convolving. The expected modes are:
            ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'constant'``.
        mode: interpolation mode for rotating the kernel. ``'bilinear'`` or ``'nearest'``.

    Return:
        the blurred image with shape :math:`(B, C, D, H, W)`.

    Example:
        >>> input = torch.randn(1, 3, 120, 80, 90).repeat(2, 1, 1, 1, 1)
        >>> # perform exact motion blur across the batch
        >>> out_1 = motion_blur3d(input, 5, (0., 90., 90.), 1)
        >>> torch.allclose(out_1[0], out_1[1])
        True
        >>> # perform element-wise motion blur across the batch
        >>> out_1 = motion_blur3d(input, 5, torch.tensor([[0., 90., 90.], [90., 180., 0.]]), torch.tensor([1., -1.]))
        >>> torch.allclose(out_1[0], out_1[1])
        False
    """
    if border_type not in ["constant", "reflect", "replicate", "circular"]:
        raise AssertionError
    kernel: Tensor = get_motion_kernel3d(kernel_size, angle, direction, mode)
    return filter3d(input, kernel, border_type)
