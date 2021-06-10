from typing import cast, Tuple, Union

import torch

from kornia.geometry.transform.affwarp import rotate, rotate3d
from kornia.utils import _extract_device_dtype


def get_motion_kernel2d(
    kernel_size: int,
    angle: Union[torch.Tensor, float],
    direction: Union[torch.Tensor, float] = 0.0,
    mode: str = 'nearest',
) -> torch.Tensor:
    r"""Return 2D motion blur filter.

    Args:
        kernel_size (int): motion kernel width and height. It should be odd and positive.
        angle (torch.Tensor, float): angle of the motion blur in degrees (anti-clockwise rotation).
        direction (float): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
        mode (str): interpolation mode for rotating the kernel. ``'bilinear'`` or ``'nearest'``.
            Default: ``'nearest'``

    Returns:
        torch.Tensor: the motion blur kernel.

    Shape:
        - Output: :math:`(B, ksize, ksize)`

    Examples::
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
        [angle if isinstance(angle, torch.Tensor) else None, direction if isinstance(direction, torch.Tensor) else None]
    )

    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size < 3:
        raise TypeError("ksize must be an odd integer >= than 3")

    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor([angle], device=device, dtype=dtype)

    angle = cast(torch.Tensor, angle)
    if angle.dim() == 0:
        angle = angle.unsqueeze(0)
    assert angle.dim() == 1, f"angle must be a 1-dim tensor. Got {angle}."

    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor([direction], device=device, dtype=dtype)

    direction = cast(torch.Tensor, direction)
    if direction.dim() == 0:
        direction = direction.unsqueeze(0)
    assert direction.dim() == 1, f"direction must be a 1-dim tensor. Got {direction}."

    assert direction.size(0) == angle.size(
        0
    ), f"direction and angle must have the same length. Got {direction} and {angle}."

    kernel_tuple: Tuple[int, int] = (kernel_size, kernel_size)
    # direction from [-1, 1] to [0, 1] range
    direction = (torch.clamp(direction, -1.0, 1.0) + 1.0) / 2.0
    # kernel = torch.zeros((direction.size(0), *kernel_tuple), device=device, dtype=dtype)

    # Element-wise linspace
    # kernel[:, kernel_size // 2, :] = torch.stack(
    #     [(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], dim=-1)
    # Alternatively
    # m = ((1 - 2 * direction)[:, None].repeat(1, kernel_size) / (kernel_size - 1))
    # kernel[:, kernel_size // 2, :] = direction[:, None].repeat(1, kernel_size) + m * torch.arange(0, kernel_size)
    k = torch.stack([(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], dim=-1)
    kernel = torch.nn.functional.pad(k[:, None], [0, 0, kernel_size // 2, kernel_size // 2, 0, 0])
    assert kernel.shape == torch.Size([direction.size(0), *kernel_tuple])
    kernel = kernel.unsqueeze(1)
    # rotate (counterclockwise) kernel by given angle
    kernel = rotate(kernel, angle, mode=mode, align_corners=True)
    kernel = kernel[:, 0]
    kernel = kernel / kernel.sum(dim=(1, 2), keepdim=True)
    return kernel


def get_motion_kernel3d(
    kernel_size: int,
    angle: Union[torch.Tensor, Tuple[float, float, float]],
    direction: Union[torch.Tensor, float] = 0.0,
    mode: str = 'nearest',
) -> torch.Tensor:
    r"""Return 3D motion blur filter.

    Args:
        kernel_size (int): motion kernel width, height and depth. It should be odd and positive.
        angle (tensor or tuple): Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
            If tensor, it must be :math:`(B, 3)`.
            If tuple, it must be (yaw, pitch, raw).
        direction (float): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
        mode (str): interpolation mode for rotating the kernel. ``'bilinear'`` or ``'nearest'``.
            Default: ``'nearest'``.

    Returns:
        torch.Tensor: the motion blur kernel.

    Shape:
        - Output: :math:`(B, kernel_size, kernel_size, kernel_size)`

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
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size < 3:
        raise TypeError(f"ksize must be an odd integer >= than 3. Got {kernel_size}.")

    device, dtype = _extract_device_dtype(
        [angle if isinstance(angle, torch.Tensor) else None, direction if isinstance(direction, torch.Tensor) else None]
    )

    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor([angle], device=device, dtype=dtype)

    angle = cast(torch.Tensor, angle)
    if angle.dim() == 1:
        angle = angle.unsqueeze(0)
    assert len(angle.shape) == 2 and angle.size(1) == 3, f"angle must be (B, 3). Got {angle}."

    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor([direction], device=device, dtype=dtype)

    direction = cast(torch.Tensor, direction)
    if direction.dim() == 0:
        direction = direction.unsqueeze(0)
    assert direction.dim() == 1, f"direction must be a 1-dim tensor. Got {direction}."

    assert direction.size(0) == angle.size(
        0
    ), f"direction and angle must have the same length. Got {direction} and {angle}."

    kernel_tuple: Tuple[int, int, int] = (kernel_size, kernel_size, kernel_size)
    # direction from [-1, 1] to [0, 1] range
    direction = (torch.clamp(direction, -1.0, 1.0) + 1.0) / 2.0
    kernel = torch.zeros((direction.size(0), *kernel_tuple), device=device, dtype=dtype)

    # Element-wise linspace
    # kernel[:, kernel_size // 2, kernel_size // 2, :] = torch.stack(
    #     [(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], dim=-1)
    k = torch.stack([(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], dim=-1)
    kernel = torch.nn.functional.pad(
        k[:, None, None], [0, 0, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, 0, 0]
    )
    assert kernel.shape == torch.Size([direction.size(0), *kernel_tuple])
    kernel = kernel.unsqueeze(1)
    # rotate (counterclockwise) kernel by given angle
    kernel = rotate3d(kernel, angle[:, 0], angle[:, 1], angle[:, 2], mode=mode, align_corners=True)
    kernel = kernel[:, 0]
    kernel = kernel / kernel.sum(dim=(1, 2, 3), keepdim=True)

    return kernel
