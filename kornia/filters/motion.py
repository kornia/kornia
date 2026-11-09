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

from typing import ClassVar

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK

from .filter import filter2d, filter3d
from .kernels_geometry import get_motion_kernel2d, get_motion_kernel3d

_VALID_BORDER = {"constant", "reflect", "replicate", "circular"}


def _scalar_params_as_tensors(
    input: torch.Tensor, angle: float | tuple[float, float, float] | torch.Tensor, direction: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # A Python-number parameter next to a tensor one is built like that tensor, on its device and in its dtype, so
    # the kernel is built from one device and one dtype, as it is from two tensors.
    if isinstance(angle, torch.Tensor) and not isinstance(direction, torch.Tensor):
        direction = torch.as_tensor(direction, device=angle.device, dtype=angle.dtype)
    elif isinstance(direction, torch.Tensor) and not isinstance(angle, torch.Tensor):
        angle = torch.as_tensor(angle, device=direction.device, dtype=direction.dtype)
    # Two Python-number parameters build the kernel in the input's floating dtype, but never below float32: a float64
    # input keeps float64 precision, while a half-precision kernel would quantise the rotation and move the
    # nearest-neighbour samples. That kernel is built on the CPU.
    # Tensor parameters keep their own device and dtype.
    dtype = torch.promote_types(input.dtype, torch.float32) if input.is_floating_point() else torch.get_default_dtype()
    if not isinstance(angle, torch.Tensor):
        angle = torch.as_tensor(angle, dtype=dtype)
    if not isinstance(direction, torch.Tensor):
        direction = torch.as_tensor(direction, dtype=dtype)
    return angle, direction


class MotionBlur(nn.Module):
    r"""Blur an image along a stored motion angle and direction (module form).

    ``blur = MotionBlur(kernel_size, angle, direction)`` stores its arguments, and ``out = blur(input)`` is
    :func:`motion_blur` on a ``(B, C, H, W)`` image.

    Convention:
        See the Convention block on :func:`~kornia.filters.motion_blur`.

    Args:
        kernel_size: width and height of the kernel, an odd integer of at least 3.
        angle: angle of the streak in degrees, anti-clockwise.
        direction: weighting of the streak in ``[-1, 1]``, as in :func:`motion_blur`.
        border_type: padding mode, as in :func:`motion_blur`.
        mode: interpolation of the rotated kernel, ``'bilinear'`` or ``'nearest'``.

    Shape:
        - Input: ``(B, C, H, W)``
        - Output: ``(B, C, H, W)``

    See Also:
        - :func:`motion_blur`: the one-call form.
        - :class:`MotionBlur3D`: the same for volumes.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> motion_blur = MotionBlur(3, 35., 0.5)
        >>> output = motion_blur(input)  # 2x4x5x7

    """

    def __init__(
        self, kernel_size: int, angle: float, direction: float, border_type: str = "reflect", mode: str = "nearest"
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction
        self.border_type = border_type
        self.mode = mode

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} (kernel_size={self.kernel_size}, "
            f"angle={self.angle}, direction={self.direction}, border_type={self.border_type}, mode={self.mode})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Blur an image along a configured two-dimensional motion direction.

        The motion-blur kernel simulates linear camera or object movement in
        the image plane. ``self.angle`` controls the direction of the blur, and
        ``self.direction`` controls whether the kernel is centered, forward-
        biased, or backward-biased along that line.

        Args:
            x: Image tensor with shape ``(B, C, H, W)``, where :math:`B`
                is the batch size, :math:`C` is the number of channels,
                :math:`H` is the height, and :math:`W` is the width.

        Returns:
            Tensor with shape ``(B, C, H, W)`` containing the directional
            blur response. The output uses the same batch, channel, and spatial
            layout as ``x``.
        """
        return motion_blur(x, self.kernel_size, self.angle, self.direction, self.border_type, mode=self.mode)


class MotionBlur3D(nn.Module):
    r"""Blur a volume along a stored motion angle and direction (module form).

    ``blur = MotionBlur3D(kernel_size, angle, direction)`` stores its arguments, and ``out = blur(input)`` is
    :func:`motion_blur3d` on a ``(B, C, D, H, W)`` volume. A scalar ``angle`` is stored as that value for all three
    components.

    Convention:
        See the Convention block on :func:`~kornia.filters.motion_blur3d`.

    Args:
        kernel_size: width, height and depth of the kernel, an odd integer of at least 3.
        angle: one axis-angle vector ``(rx, ry, rz)`` in degrees, as in :func:`motion_blur3d`. A float, a sequence of
          three floats, or a ``(B, 3)`` tensor.
        direction: weighting of the streak in ``[-1, 1]``, as in :func:`motion_blur3d`.
        border_type: padding mode, as in :func:`motion_blur3d`.
        mode: interpolation of the rotated kernel, ``'bilinear'`` or ``'nearest'``.

    Shape:
        - Input: ``(B, C, D, H, W)``
        - Output: ``(B, C, D, H, W)``

    See Also:
        - :func:`motion_blur3d`: the one-call form.
        - :class:`MotionBlur`: the same for images.

    Example:
        >>> input = torch.rand(2, 4, 5, 7, 9)
        >>> motion_blur = MotionBlur3D(3, 35., 0.5)
        >>> output = motion_blur(input)  # 2x4x5x7x9

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, -1, -1, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, -1, -1, -1, -1]
    ONNX_EXPORT_PSEUDO_SHAPE: ClassVar[list[int]] = [1, 3, 80, 80, 80]

    def __init__(
        self,
        kernel_size: int,
        angle: float | tuple[float, float, float] | list[float] | torch.Tensor,
        direction: float | torch.Tensor,
        border_type: str = "replicate",
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        KORNIA_CHECK(
            isinstance(angle, (torch.Tensor, int, float, list, tuple)),
            f"Angle should be a torch.Tensor, int, float or a sequence of floats. Got {angle}",
        )
        self.angle: tuple[float, float, float] | torch.Tensor
        if isinstance(angle, torch.Tensor):
            self.angle = angle
        elif isinstance(angle, (int, float)):
            self.angle = (angle, angle, angle)
        else:
            KORNIA_CHECK(len(angle) == 3, f"Angle sequence must have length 3. Got {len(angle)}")
            self.angle = (angle[0], angle[1], angle[2])

        self.direction = direction
        self.border_type = border_type
        self.mode = mode

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} (kernel_size={self.kernel_size}, "
            f"angle={self.angle}, direction={self.direction}, border_type={self.border_type}, mode={self.mode})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Blur a volume along a configured three-dimensional motion direction.

        This module extends motion blur from images to volumetric tensors. The
        configured kernel is applied through depth, height, and width so that
        movement can be modeled in three spatial dimensions.

        Args:
            x: Volume tensor with shape ``(B, C, D, H, W)``, where
                :math:`B` is the batch size, :math:`C` is the channel count,
                :math:`D` is the depth, :math:`H` is the height, and
                :math:`W` is the width.

        Returns:
            Tensor with shape ``(B, C, D, H, W)`` containing the blurred
            volume. The output keeps the same dimensional order as ``x``.
        """
        return motion_blur3d(x, self.kernel_size, self.angle, self.direction, self.border_type, mode=self.mode)


def motion_blur(
    input: torch.Tensor,
    kernel_size: int,
    angle: float | torch.Tensor,
    direction: float | torch.Tensor,
    border_type: str = "reflect",
    mode: str = "nearest",
) -> torch.Tensor:
    r"""Blur an image along a line at a given angle, as camera or object motion does.

    ``out = motion_blur(input, kernel_size, angle, direction)`` correlates each channel of a ``(B, C, H, W)`` image
    with the streak kernel of :func:`get_motion_kernel2d` and returns an image of the same shape.

    .. image:: _static/img/motion_blur.png

    Convention:
        - The kernel is :func:`~kornia.filters.get_motion_kernel2d`'s, correlated with the image by
          :func:`~kornia.filters.filter2d`; their Convention blocks cover ``angle``, ``direction`` (including the
          side on which the streak of a bright point is heaviest), ``mode``, the tensor shapes and the border modes.

    Args:
        input: image with shape ``(B, C, H, W)``.
        kernel_size: width and height of the kernel, an odd integer of at least 3.
        angle: angle of the streak in degrees, anti-clockwise. A float, or a ``(B,)`` tensor.
        direction: weighting of the streak in ``[-1, 1]``: ``0`` is uniform, ``1`` and ``-1`` are heaviest at opposite
          ends. A float, or a ``(B,)`` tensor. ``angle`` and ``direction`` need the same length, a float counting as
          1.
        border_type: padding mode, ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. The default
          ``'reflect'`` leaves a constant image constant. ``'constant'`` zero-pads, so pixels whose kernel reaches
          past an edge, at most ``kernel_size // 2`` from it, are pulled toward ``0``.
        mode: interpolation of the rotated kernel, ``'bilinear'`` or ``'nearest'``.

    Return:
        the blurred image with shape ``(B, C, H, W)``.

    See Also:
        - :class:`MotionBlur`: the module form.
        - :func:`motion_blur3d`: the same for volumes.
        - :func:`get_motion_kernel2d`: the kernel this blur applies.

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
    angle, direction = _scalar_params_as_tensors(input, angle, direction)
    kernel = get_motion_kernel2d(kernel_size, angle, direction, mode)
    return filter2d(input, kernel, border_type)


def motion_blur3d(
    input: torch.Tensor,
    kernel_size: int,
    angle: tuple[float, float, float] | torch.Tensor,
    direction: float | torch.Tensor,
    border_type: str = "replicate",
    mode: str = "nearest",
) -> torch.Tensor:
    r"""Blur a volume along a line at a given 3D angle, as camera or object motion does.

    ``out = motion_blur3d(input, kernel_size, angle, direction)`` correlates each channel of a ``(B, C, D, H, W)``
    volume with the streak kernel of :func:`get_motion_kernel3d` and returns a volume of the same shape.

    Convention:
        - The kernel is :func:`~kornia.filters.get_motion_kernel3d`'s, correlated with the volume by
          :func:`~kornia.filters.filter3d`; their Convention blocks cover ``angle``, ``direction``, ``mode`` and the
          border modes. With ``direction=1`` and a zero ``angle`` the streak of a bright voxel is heaviest toward
          increasing ``W``; a positive roll alone turns it toward increasing ``H``, and a positive pitch alone toward
          decreasing ``D``.

    Args:
        input: volume with shape ``(B, C, D, H, W)``.
        kernel_size: width, height and depth of the kernel, an odd integer of at least 3.
        angle: ``(yaw, pitch, roll)``, one Rodrigues axis-angle vector ``(rx, ry, rz)`` in degrees, not Euler
          angles; see :func:`~kornia.filters.get_motion_kernel3d`. A sequence of three floats, or a ``(B, 3)``
          tensor.
        direction: weighting of the streak in ``[-1, 1]``: ``0`` is uniform, ``1`` and ``-1`` are heaviest at opposite
          ends. A float, or a ``(B,)`` tensor. The first dimension of ``angle`` and ``direction`` must match,
          a float counting as 1.
        border_type: padding mode, ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. The default
          ``'replicate'`` leaves a constant volume constant at any size, including ``D = 1``. ``'reflect'`` and
          ``'circular'`` follow :func:`filter2d` along each axis. ``'constant'`` zero-pads, so voxels whose kernel reaches past a face, at most
          ``kernel_size // 2`` from it, are pulled toward ``0``.
        mode: interpolation of the rotated kernel, ``'bilinear'`` or ``'nearest'``.

    Return:
        the blurred volume with shape ``(B, C, D, H, W)``.

    See Also:
        - :class:`MotionBlur3D`: the module form.
        - :func:`motion_blur`: the same for images.
        - :func:`get_motion_kernel3d`: the kernel this blur applies.

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
    angle, direction = _scalar_params_as_tensors(input, angle, direction)
    kernel = get_motion_kernel3d(kernel_size, angle, direction, mode)
    return filter3d(input, kernel, border_type)
