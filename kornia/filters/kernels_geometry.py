# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import torch
import torch.nn.functional as F

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.core.utils import _extract_device_dtype
from kornia.geometry.transform import rotate, rotate3d

from .kernels import _check_kernel_size, _unpack_2d_ks, _unpack_3d_ks


def get_motion_kernel2d(
    kernel_size: int, angle: torch.Tensor | float, direction: torch.Tensor | float = 0.0, mode: str = "nearest"
) -> torch.Tensor:
    r"""Return 2D motion blur filter.

    Convention:
        - ``direction`` is clamped to ``[-1, 1]`` and weighs the line linearly before it is rotated: ``1`` gives
          ``[0.4, 0.3, 0.2, 0.1, 0]`` along the middle row of a size-5 kernel, heavy at the left end and 0 at the
          right, so only ``kernel_size - 1`` taps carry weight; ``-1`` mirrors it and ``0`` is uniform.
        - ``angle`` is in degrees and turns the line counter-clockwise as displayed (row 0 at the top), as
          :func:`~kornia.geometry.transform.rotate` does: at ``90`` the heavy end moves from the left to the bottom.
        - Because :func:`~kornia.filters.filter2d` correlates, the streak that the kernel draws from a bright point
          is heaviest on the side opposite the kernel's heavy end: to the point's right at ``angle=0``,
          ``direction=1``.
        - The rotation resamples the line with ``mode``: ``'nearest'`` copies each pixel from the nearest tap, so the
          number of taps changes with the angle (3 at 45 degrees, 7 at 25 for a size-5 line), and ``'bilinear'``
          spreads the weight off the line.
        - A floating tensor ``angle`` of shape :math:`(B,)` gives :math:`(B, k, k)` in its dtype. A tensor
          ``direction`` must match it in length, dtype and device; a float ``direction`` is not broadcast, so it
          raises for ``B > 1``.
        - With ``'nearest'``, each pixel copies the tap nearest to its source under the inverse rotation, picked
          with exact arithmetic: where the source falls exactly between two taps, such as at 30 or 120 degrees, the
          tie goes to the even coordinate, as ``torch.nn.functional.grid_sample`` breaks it. The kernel is the
          same on every device and dtype, for every full turn added to the angle, and with ``torch.compile``.

    Args:
        kernel_size: motion kernel width and height, an odd integer of at least 3.
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
        [angle if isinstance(angle, torch.Tensor) else None, direction if isinstance(direction, torch.Tensor) else None]
    )

    KORNIA_CHECK(isinstance(kernel_size, int), f"kernel_size must be an int. Got {type(kernel_size).__name__}")
    # TODO: add support to kernel_size as tuple or integer
    kernel_tuple = _unpack_2d_ks(kernel_size)
    _check_kernel_size(kernel_size, 2)

    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor([angle], device=device, dtype=dtype)

    if angle.dim() == 0:
        angle = angle[None]

    KORNIA_CHECK_SHAPE(angle, ["B"])

    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor([direction], device=device, dtype=dtype)

    if direction.dim() == 0:
        direction = direction[None]

    KORNIA_CHECK_SHAPE(direction, ["B"])
    KORNIA_CHECK(
        direction.size(0) == angle.size(0),
        f"direction and angle must have the same length. Got {direction.size(0)} and {angle.size(0)}.",
    )

    # direction from [-1, 1] to [0, 1] range
    direction = (torch.clamp(direction, -1.0, 1.0) + 1.0) / 2.0
    # Linearly interpolate the directional weights along the central row.
    step = (1 - 2 * direction) / (kernel_size - 1)
    positions = torch.arange(kernel_size, device=direction.device, dtype=direction.dtype)
    k = direction[:, None] + step[:, None] * positions
    kernel = F.pad(k[:, None], [0, 0, kernel_size // 2, kernel_size // 2, 0, 0])

    expected_shape = torch.Size([direction.size(0), *kernel_tuple])
    KORNIA_CHECK(kernel.shape == expected_shape, f"Kernel shape should be {expected_shape}. Got {kernel.shape}")
    kernel = kernel[:, None, ...]

    # rotate (counterclockwise) kernel by given angle
    if mode == "nearest":
        kernel = _rotate_nearest(kernel[:, 0], angle)
    else:
        kernel = rotate(kernel, angle, mode=mode, align_corners=True)
        kernel = kernel[:, 0]
    return kernel / kernel.sum(dim=(1, 2), keepdim=True)


def _rotate_nearest(kernel: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate square kernels :math:`(B, k, k)` counter-clockwise by ``angle`` degrees with nearest sampling.

    Each pixel copies the tap at its source under the inverse rotation, as ``rotate`` with ``mode='nearest'`` and
    ``align_corners=True`` does with exact arithmetic. The source is computed in float64 on the CPU and snapped to a
    :math:`2^{-20}` grid, so a source that falls between two taps is an exact tie whatever the roundoff of ``cos``
    and ``sin``, and ``torch.round`` breaks it half to even as ``grid_sample`` does. The weights are then gathered in
    the kernel's own dtype and device, so gradients still reach them.
    """
    batch_size, kernel_size = kernel.shape[0], kernel.shape[-1]
    center = (kernel_size - 1) / 2
    # move first, then cast: on MPS a single .to(device="cpu", dtype=torch.float64) returns zeros (torch 2.14)
    radians = torch.deg2rad(angle.detach().cpu().to(torch.float64) % 360.0)
    cos, sin = radians.cos()[:, None, None], radians.sin()[:, None, None]
    offsets = torch.arange(kernel_size, dtype=torch.float64) - center
    v, u = torch.meshgrid(offsets, offsets, indexing="ij")
    snap = 2.0**20
    col = (((cos * u - sin * v) * snap).round() / snap + center).round()
    row = (((sin * u + cos * v) * snap).round() / snap + center).round()
    inside = (col >= 0) & (col < kernel_size) & (row >= 0) & (row < kernel_size)
    # flat index plus one; 0 picks the zero weight padded in front, for a source outside the kernel
    source = torch.where(inside, row * kernel_size + col + 1, 0.0).long().flatten(1).to(kernel.device)
    weights = F.pad(kernel.flatten(1), [1, 0])
    kernel = weights.gather(1, source).view(batch_size, kernel_size, kernel_size)
    # nearest sampling has a zero gradient in the angle, as grid_sample gives; keep the angle in the graph with it
    return kernel * (1 + 0 * angle[:, None, None])


def get_motion_kernel3d(
    kernel_size: int,
    angle: torch.Tensor | tuple[float, float, float],
    direction: torch.Tensor | float = 0.0,
    mode: str = "nearest",
) -> torch.Tensor:
    r"""Return 3D motion blur filter.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_motion_kernel2d` for ``direction`` and ``mode``.
        ``angle`` is a Rodrigues axis-angle vector ``(rx, ry, rz)`` in degrees, as applied by
        :func:`~kornia.geometry.transform.rotate3d`: its direction is the rotation axis and its norm is the
        rotation angle. :func:`~kornia.geometry.transform.rotate3d` and the 3D motion blurs call the components
        ``yaw``, ``pitch`` and ``roll``, but they are not composed Euler rotations. The unrotated line lies along x,
        so with the default ``mode='nearest'`` yaw alone leaves the kernel unchanged (``'bilinear'`` resamples it
        off the line); a positive pitch alone turns the line's -x end, the heavy end for a positive ``direction``,
        towards +z, and a positive roll alone turns the line clockwise as displayed, the opposite of the 2d
        ``angle``.

    Args:
        kernel_size: motion kernel width, height and depth, an odd integer of at least 3.
        angle: Rodrigues axis-angle vector ``(rx, ry, rz)`` in degrees, not Euler angles.
            If tensor, it must be :math:`(B, 3)`.
            If tuple, it must be ``(rx, ry, rz)``.
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
        [angle if isinstance(angle, torch.Tensor) else None, direction if isinstance(direction, torch.Tensor) else None]
    )

    KORNIA_CHECK(isinstance(kernel_size, int), f"kernel_size must be an int. Got {type(kernel_size).__name__}")
    # TODO: add support to kernel_size as tuple or integer
    kernel_tuple = _unpack_3d_ks(kernel_size)
    _check_kernel_size(kernel_size, 2)

    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor([angle], device=device, dtype=dtype)

    if angle.dim() == 1:
        angle = angle[None]

    KORNIA_CHECK_SHAPE(angle, ["B", "3"])

    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor([direction], device=device, dtype=dtype)

    if direction.dim() == 0:
        direction = direction[None]

    KORNIA_CHECK_SHAPE(direction, ["B"])
    KORNIA_CHECK(
        direction.size(0) == angle.size(0),
        f"direction and angle must have the same batch size. Got {direction.shape} and {angle.shape}.",
    )

    # direction from [-1, 1] to [0, 1] range
    direction = (torch.clamp(direction, -1.0, 1.0) + 1.0) / 2.0
    kernel = torch.zeros((direction.size(0), *kernel_tuple), device=device, dtype=dtype)

    # Element-wise linspace
    # kernel[:, kernel_size // 2, kernel_size // 2, :] = torch.stack(
    #     [(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], dim=-1)
    k = torch.stack([(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], -1)
    kernel = F.pad(
        k[:, None, None], [0, 0, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, 0, 0]
    )

    expected_shape = torch.Size([direction.size(0), *kernel_tuple])
    KORNIA_CHECK(kernel.shape == expected_shape, f"Kernel shape should be {expected_shape}. Got {kernel.shape}")
    kernel = kernel[:, None, ...]

    # rotate (counterclockwise) kernel by given angle
    kernel = rotate3d(kernel, angle[:, 0], angle[:, 1], angle[:, 2], mode=mode, align_corners=True)
    kernel = kernel[:, 0]
    return kernel / kernel.sum(dim=(1, 2, 3), keepdim=True)
