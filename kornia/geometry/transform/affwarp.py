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

"""Rotate, shift, zoom, shear or resize images, and warp them by an affine matrix at their size.

The affine functions take ``(C, H, W)`` or ``(B, C, H, W)`` images, or ``(C, D, H, W)`` and ``(B, C, D, H, W)``
volumes for the ``3d`` forms, keep the input size and warp with :func:`~kornia.geometry.transform.warp_affine` (or
:func:`~kornia.geometry.transform.warp_affine3d`); all but ``affine`` and ``affine3d`` build the source→destination
pixel matrix themselves. Centres and shifts are pixels in ``(x, y)`` (or ``(x, y, z)``) order, rotation angles are
degrees (the shear angles of :class:`Affine` are radians), and a positive 2D angle turns counter-clockwise on
screen. The resizes take ``(*, H, W)`` and change the size with :func:`torch.nn.functional.interpolate`. A module
form stores its tensors as buffers, or as parameters when given a ``torch.nn.Parameter``, so they follow
``Module.to``.

Which function to use:

- Warp by an affine matrix at the input size: :func:`affine` ``(B, 2, 3)``, :func:`affine3d` ``(B, 3, 4)``.
  :class:`Affine` builds the 2D matrix from an angle, a translation, a scale and a shear in one module.
- Rotate about the centre: :func:`rotate` and :class:`Rotate`; a volume by an axis-angle vector: :func:`rotate3d`.
- Shift by pixels: :func:`translate` and :class:`Translate`. Zoom about the centre at the same size: :func:`scale`
  and :class:`Scale`. Shear by raw factors: :func:`shear` and :class:`Shear`.
- Change the size: to ``(h, w)`` or one side's length, :func:`resize` and :class:`Resize`; by a factor,
  :func:`rescale` and :class:`Rescale`; to the nearest multiples of a factor, :func:`resize_to_be_divisible`.
- For an output size other than the input's, warp by a matrix with :func:`~kornia.geometry.transform.warp_affine`
  or :func:`~kornia.geometry.transform.warp_perspective`.
"""

import numbers
from typing import Optional, Tuple, Union

import torch
from torch import nn

from kornia.core._compat import _emit_deprecation_warning
from kornia.core.ops import _device_literal, eye_like
from kornia.core.utils import _extract_device_dtype
from kornia.filters import gaussian_blur2d

from .imgwarp import get_affine_matrix2d, get_projective_transform, get_rotation_matrix2d, warp_affine, warp_affine3d

__all__ = [
    "Affine",
    "Rescale",
    "Resize",
    "Rotate",
    "Scale",
    "Shear",
    "Translate",
    "affine",
    "affine3d",
    "rescale",
    "resize",
    "resize_to_be_divisible",
    "rotate",
    "rotate3d",
    "scale",
    "shear",
    "translate",
]


def _register_transform_tensor(module: nn.Module, name: str, value: Optional[torch.Tensor]) -> None:
    # Preserve trainable constructor parameters; ordinary tensors follow Module.to
    # without adding new entries to existing checkpoints.
    if isinstance(value, nn.Parameter):
        module.register_parameter(name, value)
    else:
        module.register_buffer(name, value, persistent=False)


# utilities to compute affine matrices


def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the center of tensor plane for (H, W), (C, H, W) and (B, C, H, W)."""
    if not 2 <= len(tensor.shape) <= 4:
        raise AssertionError(f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}.")
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    # Filled on the device in eager mode: ``torch.tensor(..., device=cuda)`` copies from the host and waits for queued
    # kernels. TorchScript, which cannot compile ``_device_literal``, keeps the host literal.
    if torch.jit.is_scripting():
        return torch.tensor([center_x, center_y], device=tensor.device, dtype=tensor.dtype)
    return _device_literal([center_x, center_y], device=tensor.device, dtype=tensor.dtype)


def _compute_tensor_center3d(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the center of tensor plane for (D, H, W), (C, D, H, W) and (B, C, D, H, W)."""
    if not 3 <= len(tensor.shape) <= 5:
        raise AssertionError(f"Must be a 3D tensor as DHW, CDHW and BCDHW. Got {tensor.shape}.")
    depth, height, width = tensor.shape[-3:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center_z: float = float(depth - 1) / 2
    # Filled on the device in eager mode, as in ``_compute_tensor_center``.
    if torch.jit.is_scripting():
        return torch.tensor([center_x, center_y, center_z], device=tensor.device, dtype=tensor.dtype)
    return _device_literal([center_x, center_y, center_z], device=tensor.device, dtype=tensor.dtype)


def _compute_rotation_matrix(angle: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Compute a pure affine rotation matrix."""
    scale: torch.Tensor = torch.ones_like(center)
    matrix: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_rotation_matrix3d(
    yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor, center: torch.Tensor
) -> torch.Tensor:
    """Compute a pure affine rotation matrix."""
    if len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 0:
        yaw = yaw.unsqueeze(dim=0)
        pitch = pitch.unsqueeze(dim=0)
        roll = roll.unsqueeze(dim=0)

    if len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 1:
        yaw = yaw.unsqueeze(dim=1)
        pitch = pitch.unsqueeze(dim=1)
        roll = roll.unsqueeze(dim=1)

    if not (len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 2):
        raise AssertionError(f"Expected yaw, pitch, roll to be (B, 1). Got {yaw.shape}, {pitch.shape}, {roll.shape}.")

    angles: torch.Tensor = torch.cat([yaw, pitch, roll], dim=1)
    scales: torch.Tensor = torch.ones_like(yaw)
    matrix: torch.Tensor = get_projective_transform(center, angles, scales)
    return matrix


def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    """Compute affine matrix for translation."""
    matrix: torch.Tensor = eye_like(3, translation, shared_memory=False)

    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix


def _compute_scaling_matrix(scale: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Compute affine matrix for scaling."""
    angle: torch.Tensor = torch.zeros(scale.shape[:1], device=scale.device, dtype=scale.dtype)
    matrix: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_shear_matrix(shear: torch.Tensor) -> torch.Tensor:
    """Compute affine matrix for shearing."""
    matrix: torch.Tensor = eye_like(3, shear, shared_memory=False)

    shx, shy = torch.chunk(shear, chunks=2, dim=-1)
    matrix[..., 0, 1:2] += shx
    matrix[..., 1, 0:1] += shy
    return matrix


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L166


def affine(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp an image by a source→destination affine matrix, keeping its size.

    ``out = affine(tensor, matrix)`` warps ``(C, H, W)`` or ``(B, C, H, W)`` images by ``(B, 2, 3)`` source→destination
    pixel affine matrices and returns them at their own size: :func:`warp_affine` with ``dsize=(H, W)``, plus an
    unbatched image and broadcasting. One matrix serves every image of a batch, and one image, unbatched or of batch 1,
    is warped by every matrix, so it returns ``(B, C, H, W)`` for ``B > 1`` matrices. For another output size, use
    :func:`warp_affine`; to build the matrix from an angle, a shift, a scale and a shear, use :class:`Affine`.

    .. image:: _static/img/warp_affine.png

    Convention:
        See :doc:`Conventions & Pitfalls </get-started/conventions>` for the ``align_corners``
        defaults and sampling rules.

        - ``matrix`` is the source→destination **pixel** affine matrix ``(B, 2, 3)``

    Args:
        tensor: The image tensor to be warped in shapes of
            ``(C, H, W)`` or ``(B, C, H, W)``.
        matrix: The 2x3 affine transformation matrices, with shape ``(B, 2, 3)``.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The warped image with the same shape as the input, except that one image warped by ``B > 1`` matrices
        returns ``(B, C, H, W)``.

    See Also:
        - :func:`warp_affine`: the same warp with an explicit output size, for batched images.
        - :func:`affine3d`: the same for volumes, by ``(B, 3, 4)`` matrices.
        - :class:`Affine`: a module built from an angle, a translation, a scale and a shear instead of a matrix.
        - :func:`rotate`: rotate about the centre without building the matrix.

    Example:
        >>> img = torch.rand(1, 2, 3, 5)
        >>> aff = torch.eye(2, 3)[None]
        >>> out = affine(img, aff)
        >>> print(out.shape)
        torch.Size([1, 2, 3, 5])

    """
    # warping needs data in the shape of BCHW
    is_unbatched: bool = tensor.dim() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    if tensor.shape[0] == 1 and matrix.shape[0] != 1:
        tensor = tensor.expand(matrix.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3])

    # Keep a shared matrix singleton so warp_affine builds only one sampling grid.
    # Keep parameter-gradient reductions in their original order, and retain batch validation.
    if matrix.shape[0] != 1 or tensor.shape[0] == 0 or matrix.requires_grad:
        matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    height: int = tensor.shape[-2]
    width: int = tensor.shape[-1]
    warped: torch.Tensor = warp_affine(tensor, matrix, (height, width), mode, padding_mode, align_corners)

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


def affine3d(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """Warp a volume by a source→destination 3x4 affine matrix, keeping its size.

    ``out = affine3d(tensor, matrix)`` warps ``(C, D, H, W)`` or ``(B, C, D, H, W)`` volumes by ``(B, 3, 4)``
    source→destination voxel affine matrices and returns them at their own size, as :func:`warp_affine3d` with
    ``dsize=(D, H, W)`` does; one matrix serves every volume of a batch. Pass ``align_corners=True``: at the default
    ``False`` the warp mixes two voxel normalizations, so a translation of ``t`` voxels moves the volume
    ``t * size / (size - 1)`` voxels along that axis (`#4503 <https://github.com/kornia/kornia/issues/4503>`_). For
    images, use :func:`affine`.

    Convention:
        - ``matrix`` is the source→destination **pixel** affine matrix ``(B, 3, 4)``
        - align_corners: ``False`` by default (the 2D :func:`affine` defaults to ``True``)

    Args:
        tensor: The image tensor to be warped in shapes of
            ``(C, D, H, W)`` or ``(B, C, D, H, W)``.
        matrix: The affine transformation matrix with shape ``(B, 3, 4)``.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The warped image, with the shape of the input.

    See Also:
        - :func:`affine`: the 2D warp of images, whose default ``align_corners=True`` is correct.
        - :func:`warp_affine3d`: the same warp with an explicit output size.
        - :func:`rotate3d`: rotate a volume about its centre without building the matrix.

    Example:
        >>> img = torch.rand(1, 2, 4, 3, 5)
        >>> aff = torch.eye(3, 4)[None]
        >>> out = affine3d(img, aff)
        >>> print(out.shape)
        torch.Size([1, 2, 4, 3, 5])

    """
    # warping needs data in the shape of BCDHW
    is_unbatched: bool = tensor.dim() == 4
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    depth: int = tensor.shape[-3]
    height: int = tensor.shape[-2]
    width: int = tensor.shape[-1]
    warped: torch.Tensor = warp_affine3d(tensor, matrix, (depth, height, width), mode, padding_mode, align_corners)

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L185


def rotate(
    tensor: torch.Tensor,
    angle: torch.Tensor,
    center: Union[torch.Tensor, None] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Rotate an image counter-clockwise by an angle in degrees about its centre or a given point.

    ``out = rotate(tensor, angle)`` rotates ``(C, H, W)`` or ``(B, C, H, W)`` images by ``angle`` degrees, ``(B,)`` or
    one value for all, counter-clockwise as displayed, about ``center`` (``(B, 2)`` ``(x, y)`` pixels, by default the
    image centre ``((W - 1) / 2, (H - 1) / 2)``), and returns them at their size; the uncovered corners are padded
    by ``padding_mode``, except along a one-pixel axis, which the default ``align_corners=True`` repeats
    (`#6433 <https://github.com/kornia/kornia/issues/6433>`_). For a stored angle, use :class:`Rotate`; for volumes,
    :func:`rotate3d`; for the matrix, to compose or reuse it, :func:`get_rotation_matrix2d`.

    .. image:: _static/img/rotate.png

    Convention:
        - ``center`` is ``(x, y)`` in pixels, origin at top-left; defaults to the tensor center
        - positive ``angle`` rotates counter-clockwise as displayed (y-down image axes)

    Args:
        tensor: The image tensor to be warped in shapes of ``(C, H, W)`` or ``(B, C, H, W)``.
        angle: The angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        center: The center through which to rotate. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The rotated tensor with shape as input.

    See Also:
        - :class:`Rotate`: the module form, with the angle stored.
        - :func:`rotate3d`: rotate volumes by an axis-angle vector; a positive ``z`` angle turns clockwise on screen
          (3D rotations are right-handed by design).
        - :func:`get_rotation_matrix2d`: the ``(B, 2, 3)`` matrix of this rotation, with a scale.
        - :func:`rot180`: a half turn by flipping both axes, without interpolation.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/rotate_affine.html>`__.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> angle = torch.tensor([90.])
        >>> out = rotate(img, angle)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(tensor)}")

    if not isinstance(angle, torch.Tensor):
        raise TypeError(f"Input angle type is not a torch.Tensor. Got {type(angle)}")

    if center is not None and not isinstance(center, torch.Tensor):
        raise TypeError(f"Input center type is not a torch.Tensor. Got {type(center)}")

    if len(tensor.shape) not in (3, 4):
        raise ValueError(f"Invalid tensor shape, we expect CxHxW or BxCxHxW. Got: {tensor.shape}")

    # compute the rotation center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    # A singleton angle and center describe one transform shared by all images. Keep
    # it singleton through affine so warp_affine can reuse its sampling grid. Trainable
    # parameters keep the original reduction order, which matters for half-precision gradients.
    batch = tensor.shape[0] if tensor.dim() == 4 else 1
    if (
        tensor.dim() == 4
        and batch > 0
        and angle.shape == (1,)
        and not angle.requires_grad
        and not center.requires_grad
        and (center.shape == (2,) or center.shape == (1, 2))  # noqa: PLR1714 - TorchScript shape checks
        and not torch.jit.is_tracing()
    ):
        batch = 1
    angle = angle.expand(batch)
    center = center.expand(batch, -1)
    rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)

    # warp using the affine transform
    return affine(tensor, rotation_matrix[..., :2, :3], mode, padding_mode, align_corners)


def rotate3d(
    tensor: torch.Tensor,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    roll: torch.Tensor,
    center: Union[torch.Tensor, None] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """Rotate a volume by an axis-angle vector of yaw, pitch and roll in degrees.

    ``out = rotate3d(tensor, yaw, pitch, roll)`` rotates ``(C, D, H, W)`` or ``(B, C, D, H, W)`` volumes by the
    axis-angle vector ``(yaw, pitch, roll)`` in degrees, ``(B,)`` each, about ``center`` (``(B, 3)`` ``(x, y, z)``
    pixels, by default the volume centre), and returns them at their size. Positive angles follow the right-hand rule,
    so a positive ``roll``, about ``z``, turns clockwise on screen, the opposite of :func:`rotate`
    (3D rotations are right-handed by design). Pass ``align_corners=True``: at the default ``False`` the
    warp mixes two voxel normalizations, which changes the rotation unless it turns about the volume centre in a plane
    whose two sizes are equal, as any rotation of a cube about its centre does
    (`#4503 <https://github.com/kornia/kornia/issues/4503>`_).

    Convention:
        - ``center`` is ``(x, y, z)`` in pixels, origin at the top-left of the first depth
          slice (``z = 0``); defaults to the tensor center
        - ``yaw``, ``pitch`` and ``roll`` are **one axis-angle vector** ``(rx, ry, rz)`` in degrees,
          not three composed per-axis Euler rotations: the three are concatenated and handed to
          :func:`~kornia.geometry.transform.get_projective_transform`, which applies Rodrigues'
          formula. The rotation axis is ``(rx, ry, rz) / ||(rx, ry, rz)||`` and the angle is
          ``||(rx, ry, rz)||`` degrees, so ``(90, 90, 0)`` turns through ``90 * sqrt(2)`` degrees
          about ``(1, 1, 0) / sqrt(2)`` while ``(30, 0, 0)`` is a 30-degree turn about ``x``. A
          single non-zero component therefore *is* the per-axis rotation it names, which is why
          one angle reads the same under either reading; two or more do not, and neither Euler
          order reproduces the axis-angle result.
        - align_corners: ``False`` by default (the 2D :func:`rotate` defaults to ``True``)

    Args:
        tensor: The image tensor to be warped in shapes of ``(C, D, H, W)`` or ``(B, C, D, H, W)``.
        yaw: The ``rx`` component of the axis-angle vector. The tensor
          must have a shape of (B), where B is batch size.
        pitch: The ``ry`` component of the axis-angle vector. The tensor
          must have a shape of (B), where B is batch size.
        roll: The ``rz`` component of the axis-angle vector. The tensor
          must have a shape of (B), where B is batch size.
        center: The center through which to rotate. The tensor
          must have a shape of (B, 3), where B is batch size and last
          dimension contains x, y and z.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        torch.Tensor: The rotated tensor with shape as input.

    See Also:
        - :func:`rotate`: the 2D rotation of images, counter-clockwise on screen.
        - :func:`affine3d`: warp a volume by any ``(B, 3, 4)`` affine matrix.
        - :func:`get_projective_transform`: the ``(B, 3, 4)`` matrix of this rotation, with a scale.

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(tensor)}")

    if not isinstance(yaw, torch.Tensor):
        raise TypeError(f"yaw is not a torch.Tensor. Got {type(yaw)}")

    if not isinstance(pitch, torch.Tensor):
        raise TypeError(f"pitch is not a torch.Tensor. Got {type(pitch)}")

    if not isinstance(roll, torch.Tensor):
        raise TypeError(f"roll is not a torch.Tensor. Got {type(roll)}")

    if center is not None and not isinstance(center, torch.Tensor):
        raise TypeError(f"Input center type is not a torch.Tensor. Got {type(center)}")

    if len(tensor.shape) not in (4, 5):
        raise ValueError(f"Invalid tensor shape, we expect CxDxHxW or BxCxDxHxW. Got: {tensor.shape}")

    # compute the rotation center
    if center is None:
        center = _compute_tensor_center3d(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    batch = tensor.shape[0] if tensor.dim() == 5 else 1
    yaw = yaw.expand(batch)
    pitch = pitch.expand(batch)
    roll = roll.expand(batch)
    center = center.expand(batch, -1)
    rotation_matrix: torch.Tensor = _compute_rotation_matrix3d(yaw, pitch, roll, center)

    # warp using the affine transform
    return affine3d(tensor, rotation_matrix[..., :3, :4], mode, padding_mode, align_corners)


def translate(
    tensor: torch.Tensor,
    translation: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Shift an image by a translation in pixels.

    ``out = translate(tensor, translation)`` shifts ``(C, H, W)`` or ``(B, C, H, W)`` images by ``translation``
    ``(B, 2)``, ``(dx, dy)`` in pixels, a positive ``dx`` to the right and a positive ``dy`` down, and returns them at
    their size; a fractional shift interpolates, and the uncovered border is padded by ``padding_mode``, except
    along a one-pixel axis, which the default ``align_corners=True`` repeats
    (`#6433 <https://github.com/kornia/kornia/issues/6433>`_). One
    translation serves every image of a batch, and one image, unbatched or of batch 1, is shifted by every
    translation. For a stored translation, use :class:`Translate`; for the matrix, :func:`get_translation_matrix2d`.

    .. image:: _static/img/translate.png

    Convention:
        - ``translation`` is ``(dx, dy)`` in pixels

    Args:
        tensor: The image tensor to be warped in shapes of ``(C, H, W)`` or ``(B, C, H, W)``.
        translation: tensor containing the amount of pixels to
          translate in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains dx dy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The translated tensor with shape as input, except that one image shifted by ``B > 1`` translations returns
        ``(B, C, H, W)``.

    See Also:
        - :class:`Translate`: the module form, with the translation stored.
        - :func:`get_translation_matrix2d`: the ``(B, 3, 3)`` matrix of this shift.
        - :func:`affine`: warp by any affine matrix at the input size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> translation = torch.tensor([[1., 0.]])
        >>> out = translate(img, translation)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(tensor)}")

    if not isinstance(translation, torch.Tensor):
        raise TypeError(f"Input translation type is not a torch.Tensor. Got {type(translation)}")

    if len(tensor.shape) not in (3, 4):
        raise ValueError(f"Invalid tensor shape, we expect CxHxW or BxCxHxW. Got: {tensor.shape}")

    # compute the translation matrix
    translation_matrix: torch.Tensor = _compute_translation_matrix(translation)

    # warp using the affine transform
    return affine(tensor, translation_matrix[..., :2, :3], mode, padding_mode, align_corners)


def scale(
    tensor: torch.Tensor,
    scale_factor: torch.Tensor,
    center: Union[torch.Tensor, None] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Zoom an image about its centre by a scale factor, keeping its size.

    ``out = scale(tensor, scale_factor)`` scales the content of ``(C, H, W)`` or ``(B, C, H, W)`` images by
    ``scale_factor``, ``(B,)`` for both axes or ``(B, 2)`` as ``(sx, sy)``, about ``center`` (``(B, 2)`` ``(x, y)``
    pixels, by default the image centre), and returns them at their size: a factor above 1 zooms in and crops, one
    below 1 shrinks the content and pads the border, except along a one-pixel axis, which the default
    ``align_corners=True`` repeats (`#6433 <https://github.com/kornia/kornia/issues/6433>`_). To change the image
    size instead, use :func:`rescale` or :func:`resize`; for a stored factor, :class:`Scale`.

    .. image:: _static/img/scale.png

    Convention:
        - ``center`` is ``(x, y)`` in pixels, origin at top-left; defaults to the tensor center

    Args:
        tensor: The image tensor to be warped in shapes of ``(C, H, W)`` or ``(B, C, H, W)``.
        scale_factor: The scale factor apply. The tensor
          must have a shape of (B) or (B, 2), where B is batch size.
          If (B), isotropic scaling will perform.
          If (B, 2), x-y-direction specific scaling will perform.
        center: The center through which to scale. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The scaled tensor with the same shape as the input.

    See Also:
        - :class:`Scale`: the module form, with the factor stored.
        - :func:`rescale`: change the image size by a factor, given height first.
        - :func:`get_rotation_matrix2d`: the matrix of this scaling, with a zero angle.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> scale_factor = torch.tensor([[2., 2.]])
        >>> out = scale(img, scale_factor)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(tensor)}")

    if not isinstance(scale_factor, torch.Tensor):
        raise TypeError(f"Input scale_factor type is not a torch.Tensor. Got {type(scale_factor)}")

    if len(scale_factor.shape) == 1:
        # convert isotropic scaling to x-y direction
        scale_factor = scale_factor.unsqueeze(-1)

    # compute the tensor center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    batch = tensor.shape[0] if tensor.dim() == 4 else 1
    center = center.expand(batch, -1)
    scale_factor = scale_factor.expand(batch, 2)
    scaling_matrix: torch.Tensor = _compute_scaling_matrix(scale_factor, center)

    # warp using the affine transform
    return affine(tensor, scaling_matrix[..., :2, :3], mode, padding_mode, align_corners)


def shear(
    tensor: torch.Tensor,
    shear: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """Shear an image by raw shear factors about pixel (0, 0).

    ``out = shear(tensor, shear)`` shears ``(C, H, W)`` or ``(B, C, H, W)`` images by ``shear`` ``(B, 2)``, the raw
    factors ``(shx, shy)`` described below, and returns them at their size. One pair of factors serves every image of
    a batch, and one image, unbatched or of batch 1, is sheared by every pair. For shear angles in radians, use
    :class:`Affine`, about the image centre by default, or :func:`get_shear_matrix2d`, about a centre you pass; both
    shear by the negative tangent, so values below ``pi / 2`` in magnitude shear the other way there. For stored
    factors, use :class:`Shear`.

    .. image:: _static/img/shear.png

    Convention:
        - ``shear`` is a pair of raw factors ``(shx, shy)`` about pixel ``(0, 0)``:
          ``x_out = x + shx * y``, ``y_out = y + shy * x`` (x right, y down).
        - Known defect: :class:`Affine` reads its ``shear`` as angles in radians about the image centre and shears
          by their negative tangent, so the same tensor shears the other way there while its values stay below
          ``pi / 2`` in magnitude (`#5661 <https://github.com/kornia/kornia/issues/5661>`_).
        - align_corners: ``False`` by default (differs from the other 2D affine warps and from
          :class:`Shear`, which default to ``True``)

    Args:
        tensor: The image tensor to be skewed with shape of ``(C, H, W)`` or ``(B, C, H, W)``.
        shear: tensor containing raw shear factors
          in the x and y direction about pixel (0, 0). The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains shx shy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The skewed tensor with shape same as the input, except that one image sheared by ``B > 1`` pairs of
        factors returns ``(B, C, H, W)``.

    See Also:
        - :class:`Shear`: the module form, which defaults to ``align_corners=True``.
        - :func:`get_shear_matrix2d`: a shear given as angles in radians about a centre, the reading of
          :class:`Affine`.
        - :func:`affine`: warp by any affine matrix at the input size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> shear_factor = torch.tensor([[0.5, 0.0]])
        >>> out = shear(img, shear_factor)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(tensor)}")

    if not isinstance(shear, torch.Tensor):
        raise TypeError(f"Input shear type is not a torch.Tensor. Got {type(shear)}")

    if len(tensor.shape) not in (3, 4):
        raise ValueError(f"Invalid tensor shape, we expect CxHxW or BxCxHxW. Got: {tensor.shape}")

    # compute the translation matrix
    shear_matrix: torch.Tensor = _compute_shear_matrix(shear)

    # warp using the affine transform
    return affine(tensor, shear_matrix[..., :2, :3], mode, padding_mode, align_corners)


def _side_to_image_size(side_size: int, aspect_ratio: float, side: str = "short") -> Tuple[int, int]:
    if side not in ("short", "long", "vert", "horz"):
        raise ValueError(f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'")
    if side == "vert":
        return side_size, int(side_size * aspect_ratio)
    if side == "horz":
        return int(side_size / aspect_ratio), side_size
    if (side == "short") ^ (aspect_ratio < 1.0):
        return side_size, int(side_size * aspect_ratio)
    return int(side_size / aspect_ratio), side_size


def resize(
    input: torch.Tensor,
    size: Union[int, Tuple[int, int]],
    interpolation: str = "bilinear",
    align_corners: Optional[bool] = None,
    side: str = "short",
    antialias: bool = False,
) -> torch.Tensor:
    """Resize an image to a given size, or one side to a given length keeping the aspect ratio.

    ``out = resize(input, size)`` resamples the last two dimensions of ``(*, H, W)`` tensors to ``size = (h, w)`` with
    :func:`torch.nn.functional.interpolate` and returns ``(*, h, w)``. An ``int`` size sets the length of one side and
    keeps the aspect ratio, truncating the other side to an integer: the shorter side by default, or the longer side,
    the height or the width with ``side='long'``, ``'vert'`` or ``'horz'``. ``antialias=True`` blurs with a Gaussian
    before downscaling. To scale by a factor, use :func:`rescale`; to zoom the content and keep the size,
    :func:`scale`.

    .. image:: _static/img/resize.png

    Convention:
        See :doc:`Conventions & Pitfalls </get-started/conventions>` for sampling and ``align_corners``
        conventions. When resizing camera images, see
        :doc:`camera and world conventions </get-started/camera-conventions>` for matching intrinsics scaling.

        - input: ``(*, H, W)``; ``size`` is ``(h, w)``
        - align_corners: ``None`` by default (follows ``torch.nn.functional.interpolate``;
          note :func:`warp_perspective`/:func:`rotate` default ``True``)
        - ``side`` resizing preserves aspect ratio using the named side

    Args:
        input: The image tensor to resize, with shape ``(*, H, W)``; ``*`` means any number of dimensions.
        size: Desired output size. If size is a sequence like (h, w),
            output size will be matched to this. If size is an int, the side that ``side`` names will be matched to
            this number; with the default ``side='short'``, if height > width, the image will be rescaled
            to (size * height / width, size)
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'nearest-exact'`` | ``'bilinear'`` |
            ``'bicubic'`` | ``'area'``.
        align_corners: interpolation flag.
        side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
            or ``'horz'``.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The resized tensor with the shape as the specified size.

    See Also:
        - :class:`Resize`: the module form, with the size stored.
        - :func:`rescale`: the output size as a factor of the input size.
        - :func:`resize_to_be_divisible`: the nearest size whose sides are multiples of a factor.
        - :func:`scale`: zoom the content about the centre and keep the size.
        - :func:`crop_and_resize`: resamples a box of each image, not the whole image, to a fixed size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = resize(img, (6, 8))
        >>> print(out.shape)
        torch.Size([1, 3, 6, 8])

    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) < 2:
        raise ValueError(f"Input tensor must have at least two dimensions. Got {len(input.shape)}")
    # Every input is resampled as a 4-D (B, C, H, W) batch, for which F.interpolate rejects the 1-D and 3-D modes.
    if interpolation in ("linear", "trilinear"):
        raise ValueError(
            f"interpolation={interpolation!r} is not supported: resize resamples (H, W), so use 'nearest', "
            "'nearest-exact', 'bilinear', 'bicubic' or 'area'."
        )

    original_shape = input.shape
    h, w = input.shape[-2:]

    if isinstance(size, int):
        if h <= 0 or w <= 0:
            # An int size names one side and takes the other from the aspect ratio, which a
            # degenerate input does not have. Reject it with the message the warping ops use,
            # rather than letting the division below fail with a bare ZeroDivisionError.
            raise ValueError(f"Input image size must be positive. Got height={h}, width={w}.")
        aspect_ratio = w / h
        size = _side_to_image_size(size, aspect_ratio, side)
    if len(original_shape) == 2:
        # (H, W) -> (1, 1, H, W)
        input = input.unsqueeze(0).unsqueeze(0)
    elif len(original_shape) == 3:
        # (C, H, W) -> (1, C, H, W)
        input = input.unsqueeze(0)
    elif len(original_shape) > 4:
        # Flatten all leading dims: (*, C, H, W) -> (B, C, H, W)
        batch_size = 1
        for d in original_shape[:-3]:
            batch_size *= d
        input = input.reshape(batch_size, *original_shape[-3:])

    if size[0] == 0 or size[1] == 0:
        # An output side of zero gives an empty image, matching warp_affine,
        # warp_perspective and center_crop -- including when the input is empty as well,
        # which is why this runs ahead of the positive-size check below. The result is
        # built directly: the scale factors would divide by zero, and
        # torch.nn.functional.interpolate rejects a zero-sized output outright.
        #
        # It is routed through the input so the empty result keeps its autograd link, the
        # way _empty_warp_output_2d does for the warping ops. sum() promotes an integer
        # input to int64, so the accumulation dtype is pinned to the input's.
        zero = input.reshape(-1)[:1].sum(dtype=input.dtype) * 0
        output = zero.reshape(*([1] * input.dim())).expand(*input.shape[:-2], size[0], size[1])
    elif h <= 0 or w <= 0:
        raise ValueError(f"Input image size must be positive. Got height={h}, width={w}.")
    else:
        factors = (h / size[0], w / size[1])
        antialias = antialias and (max(factors) > 1)

        if antialias:
            sigmas = (max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001))

            ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

            ks = (ks[0] if ks[0] % 2 else ks[0] + 1, ks[1] if ks[1] % 2 else ks[1] + 1)
            # A singleton axis has nothing to smooth and cannot accept reflection padding.
            ks = (1 if h == 1 else ks[0], 1 if w == 1 else ks[1])

            input = gaussian_blur2d(input, ks, sigmas)

        if interpolation == "nearest-exact" and torch.onnx.is_in_onnx_export():
            # The TorchScript ONNX exporter has no symbolic for ``nearest-exact``; gather the
            # half-pixel indices ``floor((i + 0.5) * in / out)`` that ``interpolate`` samples.
            for dim, (in_size, out_size) in zip((-2, -1), ((h, size[0]), (w, size[1]))):
                index = ((torch.arange(out_size, device=input.device) + 0.5) * (in_size / out_size)).floor()
                input = input.index_select(dim, index.long().clamp(max=in_size - 1))
            output = input
        else:
            output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)

    if len(original_shape) == 2:
        output = output[0, 0]
    elif len(original_shape) == 3:
        output = output[0]
    elif len(original_shape) > 4:
        output = output.reshape(*original_shape[:-2], size[0], size[1])

    return output


def resize_to_be_divisible(
    input: torch.Tensor,
    divisible_factor: int,
    interpolation: str = "bilinear",
    align_corners: Optional[bool] = None,
    side: Optional[str] = None,
    antialias: bool = False,
) -> torch.Tensor:
    """Resize an image so that its height and width are multiples of a factor.

    ``out = resize_to_be_divisible(input, divisible_factor)`` resizes ``(*, H, W)`` tensors with :func:`resize` to the
    nearest multiples of ``divisible_factor``, at least ``divisible_factor`` for a positive side, and returns
    ``(*, h, w)``; an empty side stays empty. A side halfway between two multiples rounds as Python's :func:`round`
    does, to an even quotient, so with a factor of 4 both 6 and 10 become 8. ``side`` is deprecated and has no effect.
    For a given size, use :func:`resize`.

    Convention:
        - see the convention block of :func:`resize`
        - rounds ``height``/``width`` to the nearest multiple of ``divisible_factor`` before
          delegating to :func:`resize`; a positive side becomes at least ``divisible_factor``, and an empty side
          stays empty

    Args:
        input (torch.Tensor): Input tensor to be resized.
        divisible_factor (int): The factor to which the image should be divisible.
        interpolation (str, optional): Interpolation flag. Defaults to "bilinear".
        align_corners (Optional[bool], optional):
            whether to align the corners of the input and output. Defaults to None.
        side (str, optional): Deprecated and ignored. Both sides are rounded independently to a multiple of
            ``divisible_factor``. In eager mode, any explicit non-None value emits a :class:`DeprecationWarning`.
            Defaults to None. This argument will be removed in a future release.
        antialias (bool, optional):
            If True, then image will be filtered with Gaussian before downscaling. Defaults to False.

    Returns:
        torch.Tensor: The resized tensor.

    Raises:
        TypeError: If ``input`` is not a tensor.
        ValueError: If ``input`` has fewer than two dimensions, or ``divisible_factor`` is not a positive integer.

    See Also:
        - :func:`resize`: resize to a given size or side length.
        - :func:`rescale`: resize by a factor.

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript needs a standalone guard.
        if side is not None:
            _emit_deprecation_warning(
                "resize_to_be_divisible(..., side=...)",
                None,
                "0.9.1",
                "The `side` argument has no effect. Both sides are rounded to a multiple of `divisible_factor`.",
            )
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(input)}")
    if input.ndim < 2:
        raise ValueError(f"Input tensor must have at least two dimensions. Got {input.ndim}")
    if not isinstance(divisible_factor, numbers.Integral) or divisible_factor <= 0:
        raise ValueError(f"divisible_factor must be a positive integer. Got {divisible_factor}")

    height, width = input.shape[-2:]
    height = max(1, round(height / divisible_factor)) * divisible_factor if height > 0 else 0
    width = max(1, round(width / divisible_factor)) * divisible_factor if width > 0 else 0
    return resize(input, (height, width), interpolation=interpolation, align_corners=align_corners, antialias=antialias)


def rescale(
    input: torch.Tensor,
    factor: Union[float, Tuple[float, float]],
    interpolation: str = "bilinear",
    align_corners: Optional[bool] = None,
    antialias: bool = False,
) -> torch.Tensor:
    """Resize an image by a scale factor per side, given height first.

    ``out = rescale(input, factor)`` resizes ``(*, H, W)`` tensors with :func:`resize` to
    ``(int(H * factor_h), int(W * factor_w))``, truncating, and returns ``(*, h, w)``. ``factor`` is one ``float`` for
    both sides or a pair ``(factor_h, factor_w)``, height first, unlike the ``(sx, sy)`` of :func:`scale`; a single
    ``int`` raises ``TypeError``. To set the output size directly, use :func:`resize`; for a stored factor,
    :class:`Rescale`.

    .. image:: _static/img/rescale.png

    Convention:
        - ``factor`` is ``(factor_h, factor_w)`` when a pair — height first (contrast
          :func:`scale`, whose ``scale_factor`` is x-first)
        - delegates to :func:`resize` after converting ``factor`` to an output ``size``
        - ONNX: an export with a dynamic height or width keeps it dynamic for the ``'nearest'``,
          ``'bilinear'`` and ``'bicubic'`` modes without ``antialias``; the other modes keep the output size of
          the traced input. The exported graph computes the size in float32, which can be one pixel off eager
          when ``factor`` is not a power of two (``0.7`` at a side of 90 gives 63, eager 62)

    Args:
        input: The image tensor to be scale with shape of ``(*, H, W)``.
        factor: Desired scaling factor as ``(factor_h, factor_w)`` — height first. If a single
            float, the value is used for both height and width (an ``int`` raises ``TypeError``).
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'nearest-exact'`` | ``'bilinear'`` |
            ``'bicubic'`` | ``'area'``.
        align_corners: interpolation flag.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The rescaled tensor with the shape as the specified size.

    See Also:
        - :class:`Rescale`: the module form, which defaults to ``align_corners=True``.
        - :func:`resize`: the output size instead of a factor.
        - :func:`scale`: zoom the content by a factor and keep the size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = rescale(img, (2, 3))
        >>> print(out.shape)
        torch.Size([1, 3, 8, 12])

    """
    if isinstance(factor, float):
        factor_vert = factor_horz = factor
    else:
        factor_vert, factor_horz = factor

    if (
        torch.onnx.is_in_onnx_export()
        and not antialias
        and interpolation in ("nearest", "bilinear", "bicubic")
        and factor_vert > 0
        and factor_horz > 0
        and input.ndim == 4
    ):
        # ONNX export: keep the scale in the graph, so a dynamic height or width stays dynamic instead of freezing
        # the traced output size. ``is_exporting()`` is not used: it is also true inside ``torch.compile`` on torch
        # 2.5.1 and 2.9.1, where this path would skip ``resize``.
        return torch.nn.functional.interpolate(
            input,
            scale_factor=(factor_vert, factor_horz),
            mode=interpolation,
            align_corners=align_corners,
            recompute_scale_factor=True,
        )

    height, width = input.size()[-2:]
    size = (int(height * factor_vert), int(width * factor_horz))
    return resize(input, size, interpolation=interpolation, align_corners=align_corners, antialias=antialias)


class Resize(nn.Module):
    """Resize an image to a given size, or one side to a given length (module form).

    Build ``resizer = Resize(size)`` with the arguments of :func:`resize`, then ``out = resizer(input)`` resizes
    ``(*, H, W)`` tensors exactly as ``resize(input, size, ...)`` does. For a factor instead of a size, use
    :class:`Rescale`; inside an augmentation pipeline, use :class:`~kornia.augmentation.Resize`.

    Convention:
        See :doc:`Conventions & Pitfalls </get-started/conventions>` for sampling and ``align_corners``
        conventions. When resizing camera images, see
        :doc:`camera and world conventions </get-started/camera-conventions>` for matching intrinsics scaling.

        - See the convention block of :func:`resize`.

    Args:
        size: Desired output size. If size is a sequence like (h, w),
            output size will be matched to this. If size is an int, the side that ``side`` names will be matched to
            this number; with the default ``side='short'``, if height > width, the image will be rescaled
            to (size * height / width, size)
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'nearest-exact'`` | ``'bilinear'`` |
            ``'bicubic'`` | ``'area'``.
        align_corners: interpolation flag.
        side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
            or ``'horz'``.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The resized tensor with the shape of the given size.

    See Also:
        - :func:`resize`: the one-call form.
        - :class:`Rescale`: the module form of :func:`rescale`, a factor instead of a size.
        - :class:`~kornia.augmentation.Resize`: the augmentation of the same name, for augmentation pipelines.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = Resize((6, 8))(img)
        >>> print(out.shape)
        torch.Size([1, 3, 6, 8])

    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = "bilinear",
        align_corners: Optional[bool] = None,
        side: str = "short",
        antialias: bool = False,
    ) -> None:
        super().__init__()
        self.size: Union[int, Tuple[int, int]] = size
        self.interpolation: str = interpolation
        self.align_corners: Optional[bool] = align_corners
        self.side: str = side
        self.antialias: bool = antialias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Resize the spatial dimensions of an image or feature tensor.

        Args:
            input: Tensor with image-like layout ``(*, C, H, W)``, where
                ``*`` represents optional leading batch dimensions,
                :math:`C` is channels, :math:`H` is height, and :math:`W` is
                width.

        Returns:
            Resized tensor produced by :func:`resize`. The output keeps the
            same leading and channel dimensions as ``input``; the height and
            width are determined by ``self.size`` and ``self.side``.
        """
        return resize(
            input,
            self.size,
            self.interpolation,
            align_corners=self.align_corners,
            side=self.side,
            antialias=self.antialias,
        )


class Affine(nn.Module):
    """Warp an image by a rotation, a scale and a shear about a centre, then a translation (module form).

    Build ``warper = Affine(angle, translation, scale_factor, shear, center)`` with at least one of the first four,
    then ``out = warper(input)`` warps ``(B, C, H, W)`` images, or one ``(C, H, W)`` image, by the matrix of
    :func:`get_affine_matrix2d`: a shear, a scale, a counter-clockwise rotation in degrees and a translation in pixels,
    the first three about ``center`` (by default the image centre). The parameters, and ``center`` when given, share
    one batch size ``P``. ``P`` and the input batch must be equal, or one of them 1, a ``(C, H, W)`` input counting
    as 1: one parameter set warps every image, and ``P`` sets warp one image. For a matrix you already have, use
    :func:`affine`; for one kind of warp, :class:`Rotate`, :class:`Translate`, :class:`Scale` or :class:`Shear`.

    Convention:
        - ``shear`` contains angles in radians, using the negative-tangent, x-then-y convention
          of :func:`get_shear_matrix2d` about ``center`` (by default ``((W - 1) / 2, (H - 1) / 2)``).
        - Known defect: :func:`shear` and :class:`Shear` read their ``shear`` as raw factors about pixel ``(0, 0)``,
          without the negative tangent, so the same tensor shears the other way there while its values stay below
          ``pi / 2`` in magnitude (`#5661 <https://github.com/kornia/kornia/issues/5661>`_).
        - ``align_corners`` defaults to ``True``. See the convention block of :func:`affine`.

    Args:
        angle: Angle in degrees for counter-clockwise rotation around the center. The tensor
            must have a shape of (B), where B is the batch size.
        translation: Amount of pixels for translation in x- and y-direction. The tensor must
            have a shape of (B, 2), where B is the batch size and the last dimension contains dx and dy.
        scale_factor: Factor for scaling. The tensor must have a shape of (B,2), where B is the
            batch size and the last dimension contains scale factors for x and y direction.
        shear: Angles in radians for shearing in x- and y-direction around the center. The
            tensor must have a shape of (B, 2), where B is the batch size and the last dimension
            contains sx and sy.
        center: Transformation center in pixels. The tensor must have a shape of (B, 2), where
            B is the batch size and the last dimension contains cx and cy. Defaults to the center of image to be
            transformed.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Raises:
        RuntimeError: At construction, if not one of ``angle``, ``translation``, ``scale_factor``, or ``shear`` is
            set, or if their batch sizes differ.
        DeviceError: If ``angle``, ``translation``, ``scale_factor`` and ``shear`` are not all on one device.
        TypeCheckError: If they are on one device but do not all have one dtype. The ones not given are
            created on that device and in that dtype.
        ValueError: At the call, if ``center`` and the parameters have different batch sizes (construction does not
            check ``center``), or if the parameter batch and the input batch differ and neither is 1, a
            ``(C, H, W)`` input counting as 1.

    Returns:
        The transformed tensor with same shape as input, except that one image warped by ``P > 1`` parameter sets
        returns ``(P, C, H, W)``.

    See Also:
        - :func:`get_affine_matrix2d`: the ``(B, 3, 3)`` matrix this module warps by, whose own angle turns clockwise
          (`#4408 <https://github.com/kornia/kornia/issues/4408>`_).
        - :func:`affine`: warp by an affine matrix you already have.
        - :class:`Rotate`: a rotation alone.
        - :class:`Shear`: raw shear factors about pixel ``(0, 0)``, which shear the other way below ``pi / 2``.

    Example:
        >>> img = torch.rand(1, 2, 3, 5)
        >>> angle = 90. * torch.rand(1)
        >>> out = Affine(angle)(img)
        >>> print(out.shape)
        torch.Size([1, 2, 3, 5])

    """

    def __init__(
        self,
        angle: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
        scale_factor: Optional[torch.Tensor] = None,
        shear: Optional[torch.Tensor] = None,
        center: Optional[torch.Tensor] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        batch_sizes = [arg.size()[0] for arg in (angle, translation, scale_factor, shear) if arg is not None]
        if not batch_sizes:
            msg = (
                "Affine was created without any affine parameter. At least one of angle, translation, scale_factor, or "
                "shear has to be set."
            )
            raise RuntimeError(msg)

        batch_size = batch_sizes[0]
        if not all(other == batch_size for other in batch_sizes[1:]):
            raise RuntimeError(f"The batch sizes of the affine parameters mismatch: {batch_sizes}")

        self._batch_size = batch_size

        super().__init__()
        device, dtype = _extract_device_dtype([angle, translation, scale_factor, shear])

        if angle is None:
            angle = torch.zeros(batch_size, device=device, dtype=dtype)
        _register_transform_tensor(self, "angle", angle)

        if translation is None:
            translation = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        _register_transform_tensor(self, "translation", translation)

        if scale_factor is None:
            scale_factor = torch.ones(batch_size, 2, device=device, dtype=dtype)
        _register_transform_tensor(self, "scale_factor", scale_factor)

        _register_transform_tensor(self, "shear", shear)
        _register_transform_tensor(self, "center", center)
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Warp the input with the affine transform configured in this module.

        The transform matrix is built from this module's stored angle,
        translation, scale, optional shear, and center. The resulting matrix is
        passed to :func:`affine` for sampling.

        Args:
            input: Image tensor with shape ``(B, C, H, W)``, where
                :math:`B` is batch size, :math:`C` is channel count,
                :math:`H` is height, and :math:`W` is width.

        Returns:
            Affine-warped tensor with the input's shape, except that one image warped by ``P > 1`` parameter sets
            gives ``(P, C, H, W)``. Values sampled outside the input image are handled by ``self.padding_mode``.
        """
        if self.shear is None:
            sx = sy = None
        else:
            sx, sy = self.shear[..., 0], self.shear[..., 1]

        batch = input.shape[0] if input.dim() == 4 else 1
        params_batch = self.angle.shape[0]
        # The default center is the same for every image, so it takes the parameters' batch: a batch-1 parameter set
        # then warps every image, and B parameter sets warp one image, as with an explicit center.
        center = _compute_tensor_center(input).expand(params_batch, -1) if self.center is None else self.center

        matrix = get_affine_matrix2d(self.translation, center, self.scale_factor, -self.angle, sx=sx, sy=sy)
        if params_batch not in (batch, 1) and batch != 1:
            raise ValueError(
                f"Affine has {params_batch} parameter sets for an input batch of {batch}: the two must match, or one "
                "of them must be 1."
            )
        return affine(input, matrix[..., :2, :3], self.mode, self.padding_mode, self.align_corners)


class Rescale(nn.Module):
    """Resize an image by a stored scale factor per side, given height first (module form).

    Build ``rescaler = Rescale(factor)``, then ``out = rescaler(input)`` resizes ``(*, H, W)`` tensors as
    :func:`rescale` does, except that ``align_corners`` defaults to ``True`` here and to ``None`` in the function. For a
    size instead of a factor, use :class:`Resize`.

    Convention:
        - align_corners: ``True`` by default (differs from function :func:`rescale`, whose default
          is ``None``). It applies to the ``'bilinear'`` and ``'bicubic'`` modes only. ``'nearest'``,
          ``'nearest-exact'`` and ``'area'`` ignore it, an explicit ``align_corners=True`` included, because
          ``torch.nn.functional.interpolate`` accepts no value but ``None`` for them; the function
          :func:`rescale` raises ``ValueError`` for such a call.
        - See the convention block of :func:`rescale`.

    Args:
        factor: Desired scaling factor as ``(factor_h, factor_w)`` — height first. If a single
            float, the value is used for both height and width (an ``int`` raises ``TypeError``).
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'nearest-exact'`` | ``'bilinear'`` |
            ``'bicubic'`` | ``'area'``.
        align_corners: interpolation flag.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The rescaled tensor with the shape according to the given factor.

    See Also:
        - :func:`rescale`: the one-call form, whose ``align_corners`` defaults to ``None``.
        - :class:`Resize`: the output size instead of a factor.
        - :class:`Scale`: zoom the content by a factor and keep the size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = Rescale((2, 3))(img)
        >>> print(out.shape)
        torch.Size([1, 3, 8, 12])

    """

    def __init__(
        self,
        factor: Union[float, Tuple[float, float]],
        interpolation: str = "bilinear",
        align_corners: bool = True,
        antialias: bool = False,
    ) -> None:
        super().__init__()
        self.factor: Union[float, Tuple[float, float]] = factor
        self.interpolation: str = interpolation
        self.align_corners: Optional[bool] = align_corners
        self.antialias: bool = antialias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Resize spatial dimensions by the stored scale factor.

        Args:
            input: Tensor with shape ``(*, C, H, W)``, where ``*`` is any
                leading batch shape, :math:`C` is channels, :math:`H` is
                height, and :math:`W` is width.

        Returns:
            Tensor with the same leading and channel dimensions as ``input``.
            The output height and width are computed from ``self.factor``.
        """
        # ``align_corners`` only applies to the interpolating modes: ``F.interpolate`` rejects any value but ``None``
        # for ``nearest``, ``nearest-exact`` and ``area``, which the default ``True`` would otherwise make unusable.
        align_corners = (
            self.align_corners if self.interpolation in ("linear", "bilinear", "bicubic", "trilinear") else None
        )
        return rescale(input, self.factor, self.interpolation, align_corners=align_corners, antialias=self.antialias)


class Rotate(nn.Module):
    """Rotate an image counter-clockwise by a stored angle in degrees (module form).

    Build ``rotator = Rotate(angle)``, optionally with ``center``, then ``out = rotator(input)`` rotates ``(C, H, W)``
    or ``(B, C, H, W)`` images as :func:`rotate` does: counter-clockwise as displayed, about the image centre by
    default, at the input size. For volumes, use :func:`rotate3d`; to rotate, shift, scale and shear in one warp,
    :class:`Affine`.

    Convention:
        - See the convention block of :func:`rotate`.

    Args:
        angle: The angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        center: The center through which to rotate. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The rotated tensor with the same shape as the input.

    See Also:
        - :func:`rotate`: the one-call form.
        - :class:`Affine`: a rotation with a translation, a scale and a shear in one warp.
        - :class:`~kornia.geometry.transform.Rot180`: a half turn by flipping both axes, without interpolation.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> angle = torch.tensor([90.])
        >>> out = Rotate(angle)(img)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])

    """

    def __init__(
        self,
        angle: torch.Tensor,
        center: Union[torch.Tensor, None] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        _register_transform_tensor(self, "angle", angle)
        _register_transform_tensor(self, "center", center)
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Rotate an image batch around the configured center.

        Args:
            input: Image tensor with shape ``(B, C, H, W)``, where
                :math:`B` is batch size, :math:`C` is channels, :math:`H` is
                height, and :math:`W` is width.

        Returns:
            Rotated tensor with shape ``(B, C, H, W)``. The image extent is
            kept fixed; newly exposed pixels are filled according to
            ``self.padding_mode``.
        """
        return rotate(input, self.angle, self.center, self.mode, self.padding_mode, self.align_corners)


class Translate(nn.Module):
    """Shift an image by a stored translation in pixels (module form).

    Build ``shifter = Translate(translation)``, then ``out = shifter(input)`` shifts ``(C, H, W)`` or ``(B, C, H, W)``
    images by the ``(B, 2)`` translation ``(dx, dy)`` in pixels, as :func:`translate` does. To combine the shift with a
    rotation, a scale and a shear, use :class:`Affine`.

    Convention:
        - See the convention block of :func:`translate`.

    Args:
        translation: tensor containing the amount of pixels to
          translate in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains dx dy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The translated tensor with the same shape as the input, except that one image shifted by ``B > 1``
        translations returns ``(B, C, H, W)``.

    See Also:
        - :func:`translate`: the one-call form.
        - :class:`Affine`: a translation with a rotation, a scale and a shear in one warp.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> translation = torch.tensor([[1., 0.]])
        >>> out = Translate(translation)(img)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])

    """

    def __init__(
        self,
        translation: torch.Tensor,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        _register_transform_tensor(self, "translation", translation)
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Shift an image batch by the stored pixel translation.

        Args:
            input: Image tensor with shape ``(B, C, H, W)``, where
                :math:`B` is batch size, :math:`C` is channels, :math:`H` is
                height, and :math:`W` is width.

        Returns:
            Translated tensor with shape ``(B, C, H, W)``. Translation is
            measured in pixels along the x and y image axes.
        """
        return translate(input, self.translation, self.mode, self.padding_mode, self.align_corners)


class Scale(nn.Module):
    """Zoom an image about its centre by a stored scale factor, keeping its size (module form).

    Build ``zoomer = Scale(scale_factor)``, optionally with ``center``, then ``out = zoomer(input)`` scales the content
    of ``(C, H, W)`` or ``(B, C, H, W)`` images as :func:`scale` does, at the input size. To change the image size
    instead, use :class:`Rescale` or :class:`Resize`.

    Convention:
        - See the convention block of :func:`scale`.

    Args:
        scale_factor: The scale factor apply. The tensor
          must have a shape of (B) or (B, 2), where B is batch size.
          If (B), isotropic scaling will perform.
          If (B, 2), x-y-direction specific scaling will perform.
        center: The center through which to scale. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The scaled tensor with the same shape as the input.

    See Also:
        - :func:`scale`: the one-call form.
        - :class:`Rescale`: change the image size by a factor instead.
        - :class:`Affine`: a scale with a rotation, a translation and a shear in one warp.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> scale_factor = torch.tensor([[2., 2.]])
        >>> out = Scale(scale_factor)(img)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])

    """

    def __init__(
        self,
        scale_factor: torch.Tensor,
        center: Union[torch.Tensor, None] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        _register_transform_tensor(self, "scale_factor", scale_factor)
        _register_transform_tensor(self, "center", center)
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Scale an image batch around the configured center point.

        Args:
            input: Image tensor with shape ``(B, C, H, W)``, where
                :math:`B` is batch size, :math:`C` is channels, :math:`H` is
                height, and :math:`W` is width.

        Returns:
            Scaled tensor with the same shape as ``input``. The sampling grid
            is built from ``self.scale_factor`` and ``self.center``.
        """
        return scale(input, self.scale_factor, self.center, self.mode, self.padding_mode, self.align_corners)


class Shear(nn.Module):
    """Shear an image by stored raw shear factors about pixel (0, 0) (module form).

    Build ``shearer = Shear(shear)``, then ``out = shearer(input)`` shears ``(C, H, W)`` or ``(B, C, H, W)`` images by
    the ``(B, 2)`` raw factors ``(shx, shy)``, as :func:`shear` does, except that ``align_corners`` defaults to
    ``True`` here and to ``False`` in the function. For shear angles in radians about the image centre, use
    :class:`Affine`, which shears by their negative tangent, so values below ``pi / 2`` in magnitude shear the other
    way there.

    Convention:
        - align_corners: ``True`` by default (differs from function :func:`shear`, whose default is
          ``False``)
        - See the convention block of :func:`shear`.

    Args:
        shear: tensor containing raw shear factors
          in the x and y direction about pixel (0, 0). The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains shx shy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The skewed tensor with the same shape as the input, except that one image sheared by ``B > 1`` pairs of
        factors returns ``(B, C, H, W)``.

    See Also:
        - :func:`shear`: the one-call form, which defaults to ``align_corners=False``.
        - :class:`Affine`: shear angles in radians about the image centre, which shear the other way below
          ``pi / 2``.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> shear_factor = torch.tensor([[0.5, 0.0]])
        >>> out = Shear(shear_factor)(img)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])

    """

    def __init__(
        self, shear: torch.Tensor, mode: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = True
    ) -> None:
        super().__init__()
        _register_transform_tensor(self, "shear", shear)
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Skew an image batch with the configured x/y shear factors.

        Args:
            input: Image tensor with shape ``(B, C, H, W)``, where
                :math:`B` is batch size, :math:`C` is channels, :math:`H` is
                height, and :math:`W` is width.

        Returns:
            Sheared tensor with shape ``(B, C, H, W)``. The output image
            size is unchanged; pixels outside the sampled area use the
            configured padding mode.
        """
        return shear(input, self.shear, self.mode, self.padding_mode, self.align_corners)
