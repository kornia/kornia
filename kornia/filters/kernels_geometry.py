from __future__ import annotations

import torch

from kornia.core import Tensor, pad, stack, tensor, zeros
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.geometry.transform import rotate, rotate3d
from kornia.utils import _extract_device_dtype

from .kernels import _check_kernel_size, _unpack_2d_ks, _unpack_3d_ks


def get_motion_kernel2d(
    kernel_size: int, angle: Tensor | float, direction: Tensor | float = 0.0, mode: str = "nearest"
) -> Tensor:
    r"""Return 2D motion blur filter.

    Args:
        kernel_size: motion kernel width and height. It should be odd and positive.
        angle: angle of the motion blur in degrees (anti-clockwise rotation).
        direction: forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
        mode: interpolation mode for rotating the kernel. ``'bilinear'`` or ``'nearest'``.

    Returns:
        The motion blur kernel of shape :math:`(B, k_\text{size}, k_\text{size})`.

    Examples:
        >>> get_motion_kernel2d(5, 0., 0.)
        tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                 [0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]])

        >>> get_motion_kernel2d(3, 215., -0.5)
        tensor([[[0.0000, 0.0000, 0.1667],
                 [0.0000, 0.3333, 0.0000],
                 [0.5000, 0.0000, 0.0000]]])
    """
    device, dtype = _extract_device_dtype(
        [angle if isinstance(angle, Tensor) else None, direction if isinstance(direction, Tensor) else None]
    )

    # TODO: add support to kernel_size as tuple or integer
    kernel_tuple = _unpack_2d_ks(kernel_size)
    _check_kernel_size(kernel_size, 2)

    if not isinstance(angle, Tensor):
        angle = tensor([angle], device=device, dtype=dtype)

    if angle.dim() == 0:
        angle = angle[None]

    KORNIA_CHECK_SHAPE(angle, ["B"])

    if not isinstance(direction, Tensor):
        direction = tensor([direction], device=device, dtype=dtype)

    if direction.dim() == 0:
        direction = direction[None]

    KORNIA_CHECK_SHAPE(direction, ["B"])
    KORNIA_CHECK(
        direction.size(0) == angle.size(0),
        f"direction and angle must have the same length. Got {direction} and {angle}.",
    )

    # direction from [-1, 1] to [0, 1] range
    direction = (torch.clamp(direction, -1.0, 1.0) + 1.0) / 2.0
    # kernel = torch.zeros((direction.size(0), *kernel_tuple), device=device, dtype=dtype)

    # Element-wise linspace
    # kernel[:, kernel_size // 2, :] = torch.stack(
    #     [(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], dim=-1)
    # Alternatively
    # m = ((1 - 2 * direction)[:, None].repeat(1, kernel_size) / (kernel_size - 1))
    # kernel[:, kernel_size // 2, :] = direction[:, None].repeat(1, kernel_size) + m * torch.arange(0, kernel_size)
    k = stack([(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], -1)
    kernel = pad(k[:, None], [0, 0, kernel_size // 2, kernel_size // 2, 0, 0])

    expected_shape = torch.Size([direction.size(0), *kernel_tuple])
    KORNIA_CHECK(kernel.shape == expected_shape, f"Kernel shape should be {expected_shape}. Gotcha {kernel.shape}")
    kernel = kernel[:, None, ...]

    # rotate (counterclockwise) kernel by given angle
    kernel = rotate(kernel, angle, mode=mode, align_corners=True)
    kernel = kernel[:, 0]
    kernel = kernel / kernel.sum(dim=(1, 2), keepdim=True)
    return kernel


def get_motion_kernel3d(
    kernel_size: int, angle: Tensor | tuple[float, float, float], direction: Tensor | float = 0.0, mode: str = "nearest"
) -> Tensor:
    r"""Return 3D motion blur filter.

    Args:
        kernel_size: motion kernel width, height and depth. It should be odd and positive.
        angle: Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
            If tensor, it must be :math:`(B, 3)`.
            If tuple, it must be (yaw, pitch, raw).
        direction: forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
        mode: interpolation mode for rotating the kernel. ``'bilinear'`` or ``'nearest'``.

    Returns:
        The motion blur kernel with shape :math:`(B, k_\text{size}, k_\text{size}, k_\text{size})`.

    Examples:
        >>> get_motion_kernel3d(3, (0., 0., 0.), 0.)
        tensor([[[[0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 0.0000],
                  [0.3333, 0.3333, 0.3333],
                  [0.0000, 0.0000, 0.0000]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])

        >>> get_motion_kernel3d(3, (90., 90., 0.), -0.5)
        tensor([[[[0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000],
                  [0.0000, 0.5000, 0.0000]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 0.0000],
                  [0.0000, 0.3333, 0.0000],
                  [0.0000, 0.0000, 0.0000]],
        <BLANKLINE>
                 [[0.0000, 0.1667, 0.0000],
                  [0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])
    """
    device, dtype = _extract_device_dtype(
        [angle if isinstance(angle, Tensor) else None, direction if isinstance(direction, Tensor) else None]
    )

    # TODO: add support to kernel_size as tuple or integer
    kernel_tuple = _unpack_3d_ks(kernel_size)
    _check_kernel_size(kernel_size, 2)

    if not isinstance(angle, Tensor):
        angle = tensor([angle], device=device, dtype=dtype)

    if angle.dim() == 1:
        angle = angle[None]

    KORNIA_CHECK_SHAPE(angle, ["B", "3"])

    if not isinstance(direction, Tensor):
        direction = tensor([direction], device=device, dtype=dtype)

    if direction.dim() == 0:
        direction = direction[None]

    KORNIA_CHECK_SHAPE(direction, ["B"])
    KORNIA_CHECK(
        direction.size(0) == angle.size(0),
        f"direction and angle must have the same batch size. Got {direction.shape} and {angle.shape}.",
    )

    # direction from [-1, 1] to [0, 1] range
    direction = (torch.clamp(direction, -1.0, 1.0) + 1.0) / 2.0
    kernel = zeros((direction.size(0), *kernel_tuple), device=device, dtype=dtype)

    # Element-wise linspace
    # kernel[:, kernel_size // 2, kernel_size // 2, :] = torch.stack(
    #     [(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], dim=-1)
    k = stack([(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], -1)
    kernel = pad(k[:, None, None], [0, 0, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, 0, 0])

    expected_shape = torch.Size([direction.size(0), *kernel_tuple])
    KORNIA_CHECK(kernel.shape == expected_shape, f"Kernel shape should be {expected_shape}. Gotcha {kernel.shape}")
    kernel = kernel[:, None, ...]

    # rotate (counterclockwise) kernel by given angle
    kernel = rotate3d(kernel, angle[:, 0], angle[:, 1], angle[:, 2], mode=mode, align_corners=True)
    kernel = kernel[:, 0]
    kernel = kernel / kernel.sum(dim=(1, 2, 3), keepdim=True)

    return kernel
