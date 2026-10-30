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

"""Convert between homogeneous points, normalized coordinates, rotations and poses.

Points and vectors sit in the last dimension, and matrices act on column vectors from the left. 2D pixel coordinates
are ``(x, y)``, with ``x`` the column and pixel centres at integers, while size arguments are ``(height, width)`` or
``(depth, height, width)``. The :math:`[-1, 1]` normalizations are corner-aligned, mapping the first and last pixel
centre of an axis to ``-1`` and ``1``; only :func:`normal_transform_pixel`, :func:`normalize_homography` and
:func:`denormalize_homography` also offer the half-pixel convention of ``align_corners=False``, and their 3D forms
do not (`#4503 <https://github.com/kornia/kornia/issues/4503>`_). Quaternions are ``(w, x, y, z)``, real part first,
in the Hamilton convention. 3D rotations are active and right-handed: a positive angle about ``+z`` turns ``+x``
towards ``+y``. Axis-angle vectors and Euler angles are in radians, while :func:`angle_to_rotation_matrix` takes
degrees and turns ``+x`` towards ``-y``, counter-clockwise as displayed with ``y`` down. The pose functions take
exactly one batch dimension: ``(B, 3, 3)`` rotations with ``(B, 3, 1)`` translations, ``(B, 4, 4)`` matrices or
``(B, 4)`` quaternions.

Which function to use:

- Homogeneous points: :func:`convert_points_to_homogeneous` appends ``w = 1`` to ``(*, N, D)`` points and
  :func:`convert_points_from_homogeneous` divides by it. An affine matrix as a homography:
  :func:`convert_affinematrix_to_homography` for ``(B, 2, 3)``, :func:`convert_affinematrix_to_homography3d` for
  ``(B, 3, 4)``.
- Pixel coordinates to :math:`[-1, 1]` and back: :func:`normalize_pixel_coordinates` and
  :func:`denormalize_pixel_coordinates` for ``(*, 2)`` points, :func:`normalize_pixel_coordinates3d` and
  :func:`denormalize_pixel_coordinates3d` for ``(*, 3)`` points in ``(d, x, y)`` order. The same maps as matrices:
  :func:`normal_transform_pixel` ``(1, 3, 3)`` on ``(x, y, 1)`` and :func:`normal_transform_pixel3d` ``(1, 4, 4)`` on
  ``(x, y, z, 1)``.
- A pixel homography in :math:`[-1, 1]` coordinates and back: :func:`normalize_homography` and
  :func:`denormalize_homography` for ``(B, 3, 3)``, :func:`normalize_homography3d` and
  :func:`denormalize_homography3d` for ``(B, 4, 4)``.
- Pixels to normalized camera coordinates with intrinsics ``K``, and back: :func:`normalize_points_with_intrinsics`,
  :func:`denormalize_points_with_intrinsics`.
- A 3D rotation from one form to another, axis-angle ``(*, 3)``, quaternion ``(*, 4)`` or matrix ``(*, 3, 3)``:
  :func:`axis_angle_to_rotation_matrix`, :func:`axis_angle_to_quaternion`, :func:`quaternion_to_rotation_matrix`,
  :func:`quaternion_to_axis_angle`, :func:`rotation_matrix_to_quaternion`, :func:`rotation_matrix_to_axis_angle`.
  The deprecated names ``angle_axis_to_rotation_matrix``, ``angle_axis_to_quaternion``, ``quaternion_to_angle_axis``
  and ``rotation_matrix_to_angle_axis`` forward to them.
- Quaternion helpers: unit length, :func:`normalize_quaternion`; the exponential and logarithm maps, whose 3-vector
  is half the axis-angle vector, :func:`quaternion_log_to_exp` and :func:`quaternion_exp_to_log`; roll, pitch and
  yaw, with the quaternion as four separate tensors, :func:`quaternion_from_euler` and :func:`euler_from_quaternion`.
- A 2D rotation matrix from an angle in degrees: :func:`angle_to_rotation_matrix`. The cross-product matrix of a
  3-vector: :func:`vector_to_skew_symmetric_matrix`.
- Poses: pack and split ``(B, 4, 4)`` matrices with :func:`Rt_to_matrix4x4` and :func:`matrix4x4_to_Rt`; invert a
  rigid ``(R, t)`` pose with :func:`camtoworld_to_worldtocam_Rt` or :func:`worldtocam_to_camtoworld_Rt`; switch a
  camera-to-world pose between OpenGL and OpenCV camera axes with :func:`camtoworld_graphics_to_vision_4x4`,
  :func:`camtoworld_graphics_to_vision_Rt`, :func:`camtoworld_vision_to_graphics_4x4` or
  :func:`camtoworld_vision_to_graphics_Rt`; convert quaternion poses between ARKit and Colmap with
  :func:`ARKitQTVecs_to_ColmapQTVecs` and :func:`ColmapQTVecs_to_ARKitQTVecs`.
- Angles and polar coordinates: :func:`deg2rad`, :func:`rad2deg`, :func:`cart2pol`, :func:`pol2cart`.
- Rotations and poses as objects that compose and invert: :class:`~kornia.geometry.quaternion.Quaternion`,
  :class:`~kornia.geometry.liegroup.So3` and :class:`~kornia.geometry.liegroup.Se3`.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F

from kornia.core._compat import deprecated
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE, ShapeError, are_checks_enabled
from kornia.core.exceptions import TypeCheckError
from kornia.core.ops import _constant_tensor
from kornia.core.utils import _inverse_3x3_closed_form, _torch_inverse_cast, is_compiling

__all__ = [
    "ARKitQTVecs_to_ColmapQTVecs",
    "ColmapQTVecs_to_ARKitQTVecs",
    "Rt_to_matrix4x4",
    "angle_axis_to_quaternion",
    "angle_axis_to_rotation_matrix",
    "angle_to_rotation_matrix",
    "axis_angle_to_quaternion",
    "axis_angle_to_rotation_matrix",
    "camtoworld_graphics_to_vision_4x4",
    "camtoworld_graphics_to_vision_Rt",
    "camtoworld_to_worldtocam_Rt",
    "camtoworld_vision_to_graphics_4x4",
    "camtoworld_vision_to_graphics_Rt",
    "cart2pol",
    "convert_affinematrix_to_homography",
    "convert_affinematrix_to_homography3d",
    "convert_points_from_homogeneous",
    "convert_points_to_homogeneous",
    "deg2rad",
    "denormalize_homography",
    "denormalize_homography3d",
    "denormalize_pixel_coordinates",
    "denormalize_pixel_coordinates3d",
    "denormalize_points_with_intrinsics",
    "euler_from_quaternion",
    "matrix4x4_to_Rt",
    "normal_transform_pixel",
    "normal_transform_pixel3d",
    "normalize_homography",
    "normalize_homography3d",
    "normalize_pixel_coordinates",
    "normalize_pixel_coordinates3d",
    "normalize_points_with_intrinsics",
    "normalize_quaternion",
    "pol2cart",
    "quaternion_exp_to_log",
    "quaternion_from_euler",
    "quaternion_log_to_exp",
    "quaternion_to_angle_axis",
    "quaternion_to_axis_angle",
    "quaternion_to_rotation_matrix",
    "rad2deg",
    "rotation_matrix_to_angle_axis",
    "rotation_matrix_to_axis_angle",
    "rotation_matrix_to_quaternion",
    "vector_to_skew_symmetric_matrix",
    "worldtocam_to_camtoworld_Rt",
]


def rad2deg(tensor: torch.Tensor) -> torch.Tensor:
    r"""Convert angles from radians to degrees, elementwise.

    ``degrees = rad2deg(radians)`` multiplies a tensor of any shape by ``180 / pi`` and returns the angles in degrees
    with the same shape, device and floating dtype; an integer tensor gives the default floating dtype. For the
    reverse, use :func:`deg2rad`.

    Args:
        tensor: angles in radians, of any shape.

    Returns:
        the angles in degrees, with the shape of ``tensor``.

    See Also:
        - :func:`deg2rad`: the reverse conversion, degrees to radians.

    Example:
        >>> input = torch.tensor(3.1415926535)
        >>> rad2deg(input)
        tensor(180.)

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    return tensor * (180.0 / math.pi)


def deg2rad(tensor: torch.Tensor) -> torch.Tensor:
    r"""Convert angles from degrees to radians, elementwise.

    ``radians = deg2rad(degrees)`` multiplies a tensor of any shape by ``pi / 180`` and returns the angles in radians
    with the same shape, device and floating dtype; an integer tensor gives the default floating dtype. The axis-angle,
    Euler and polar functions of this module take radians; :func:`angle_to_rotation_matrix` takes degrees. For the
    reverse, use :func:`rad2deg`.

    Args:
        tensor: angles in degrees, of any shape.

    Returns:
        the angles in radians, with the shape of ``tensor``.

    See Also:
        - :func:`rad2deg`: the reverse conversion, radians to degrees.
        - :func:`angle_to_rotation_matrix`: the 2D rotation matrix of an angle given in degrees.

    Examples:
        >>> input = torch.tensor(180.)
        >>> deg2rad(input)
        tensor(3.1416)

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    return tensor * (math.pi / 180.0)


def pol2cart(rho: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert polar coordinates (rho, phi), phi in radians, to cartesian coordinates (x, y).

    ``x, y = pol2cart(rho, phi)`` takes radii and angles in radians and returns two tensors, ``x = rho * cos(phi)`` and
    ``y = rho * sin(phi)``, in the broadcast shape of ``rho`` and ``phi``. It undoes :func:`cart2pol`:
    ``pol2cart(*cart2pol(x, y))`` returns ``(x, y)`` to rounding as long as ``x ** 2 + y ** 2`` is above the ``eps``
    of :func:`cart2pol` and neither overflows nor underflows in the working dtype. Inside the ``eps`` neighbourhood of
    the origin :func:`cart2pol` rounds the radius, so in ``float32`` ``(1e-5, 2e-5)`` comes back as
    ``(1.0023e-5, 2.0045e-5)`` and ``(3e-8, 4e-8)`` as ``(0, 0)``
    (`#6440 <https://github.com/kornia/kornia/issues/6440>`_).

    Convention:
        - ``rho = 5``, ``phi = atan2(4, 3)`` gives ``(3., 4.)``, and ``rho = 2``, ``phi = pi / 6`` gives
          ``(1.7321, 1.)``
        - ``phi`` follows the angle convention documented on :func:`cart2pol`

    Args:
        rho: the radii, of any shape that broadcasts with ``phi``.
        phi: the angles in radians.

    Returns:
        - x: the first coordinate, in the broadcast shape of ``rho`` and ``phi``.
        - y: the second coordinate, in the same shape.

    See Also:
        - :func:`cart2pol`: the reverse conversion, cartesian to polar.

    Example:
        >>> rho = torch.rand(1, 3, 3)
        >>> phi = torch.rand(1, 3, 3)
        >>> x, y = pol2cart(rho, phi)

    """
    if not (isinstance(rho, torch.Tensor) & isinstance(phi, torch.Tensor)):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rho)}, {type(phi)}")

    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return x, y


def cart2pol(x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-8) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert cartesian coordinates (x, y) to polar coordinates (rho, phi), phi in radians.

    ``rho, phi = cart2pol(x, y)`` returns two tensors in the broadcast shape of ``x`` and ``y``: the radius
    ``sqrt(x ** 2 + y ** 2)`` and the angle ``atan2(y, x)`` in radians, in :math:`[-\pi, \pi]`. The radius is
    computed from the unscaled ``x ** 2 + y ** 2``, so it is ``nan`` once that overflows
    (`#6414 <https://github.com/kornia/kornia/issues/6414>`_). ``eps`` keeps the gradient of ``rho`` finite at the
    origin, but it also rounds every radius with ``x ** 2 + y ** 2 <= eps``, as the convention below details
    (`#6440 <https://github.com/kornia/kornia/issues/6440>`_). For the reverse, use :func:`pol2cart`.

    Convention:
        - ``(3., 4.)`` gives ``(5., 0.9273)``
        - ``x ** 2 + y ** 2`` overflows from a radius of about ``256`` in
          ``float16`` (``(300., 150.)`` gives ``rho = nan``), ``1.8e19`` in
          ``float32`` and ``bfloat16`` and ``1.3e154`` in ``float64``, and
          ``rho`` is then ``nan``, not ``inf``
          (`#6414 <https://github.com/kornia/kornia/issues/6414>`_)
        - where ``x ** 2 + y ** 2 <= eps`` (radii up to about ``1e-4`` at the
          default ``eps``), ``rho`` is computed as ``(rho + 1) - 1``, which
          rounds it to a multiple of the spacing of the dtype at ``1``
          (`#6440 <https://github.com/kornia/kornia/issues/6440>`_): in
          ``float32`` ``(1e-5, 0.)`` gives ``1.0014e-5`` and every radius below
          about ``6e-8``, ``(1e-20, 0.)`` included, gives ``0``; in ``bfloat16``
          every radius up to ``1e-4`` gives ``0``; in ``float64`` ``(1e-14, 0.)``
          gives ``9.992e-15``. In ``float16`` no nonzero sum is that small, so
          only a sum of ``0`` takes that branch; there a sum that underflows
          makes ``rho`` inexact or ``0`` instead: ``(2e-4, 0.)`` gives
          ``2.44e-4``
        - ``phi`` is ``atan2(y, x)`` in **radians**, so it lies in
          :math:`[-\pi, \pi]`, is ``0`` on the ``+x`` axis and grows toward
          ``+y`` (``x = 0``, ``y = 1`` gives ``phi = 1.5708``). Both endpoints
          are attained on the ``-x`` axis, where the sign of ``y``'s zero
          decides between ``-pi`` and ``+pi``
        - kornia's image ``y`` axis points down, so a growing ``phi`` turns
          **clockwise as displayed**. The relation to the other 2-D
          angle op is the opposite sense: applying
          ``angle_to_rotation_matrix(theta)`` to a nonzero point *decreases*
          that point's ``phi`` by ``theta`` degrees **modulo** :math:`2\pi` —
          the result is re-wrapped into :math:`[-\pi, \pi]`, so a point at
          ``phi = -170`` degrees rotated by ``theta = 30`` returns
          ``phi = +160``, not ``-200``. At the origin ``phi`` carries no
          direction and the relation does not apply

    Args:
        x: the first coordinates, of any shape that broadcasts with ``y``.
        y: the second coordinates.
        eps: squared-radius threshold: at or below it the gradient of ``rho``
            is zero, which avoids the ``0 / 0`` of the exact gradient at the
            origin, and ``rho`` is rounded as described above
            (`#6440 <https://github.com/kornia/kornia/issues/6440>`_). The
            default ``1e-8`` underflows in ``float16``.

    Returns:
        - rho: the radius, in the broadcast shape of ``x`` and ``y``.
        - phi: the angle in radians, in the same shape.

    See Also:
        - :func:`pol2cart`: the reverse conversion, polar to cartesian.
        - :func:`angle_to_rotation_matrix`: the 2D rotation by an angle in degrees, which turns ``phi`` the other way.

    Example:
        >>> x = torch.rand(1, 3, 3)
        >>> y = torch.rand(1, 3, 3)
        >>> rho, phi = cart2pol(x, y)

    """
    if not (isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor)):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(x)}, {type(y)}")

    in_dtype = torch.promote_types(x.dtype, y.dtype)
    half = in_dtype in (torch.float16, torch.bfloat16)
    # float32 holds the square of every float16 value, so half-precision radii neither overflow nor lose small values
    xw, yw = (x.float(), y.float()) if half else (x, y)
    squared_radius = xw**2 + yw**2
    outside = squared_radius > eps
    # inside eps, the exact radius from a detached root keeps the zero gradient at the origin
    safe_rho = torch.sqrt(torch.where(outside, squared_radius, torch.ones_like(squared_radius)))
    rho = torch.where(outside, safe_rho, torch.sqrt(squared_radius).detach())
    if half:
        rho = rho.to(in_dtype)
    phi = torch.atan2(y, x)
    return rho, phi


def convert_points_from_homogeneous(points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert homogeneous points ``(*, N, D)`` to Euclidean ``(*, N, D - 1)``.

    ``points = convert_points_from_homogeneous(points_h)`` takes points ``(*, N, D)`` whose **last** coordinate is the
    homogeneous ``w``, divides the others by it and returns ``(*, N, D - 1)``. A point with ``abs(w) <= eps`` comes
    back undivided rather than as ``inf``, following OpenCV. For the reverse, use :func:`convert_points_to_homogeneous`;
    to apply a homography to Euclidean points in one call, :func:`~kornia.geometry.linalg.transform_points`.

    Convention:
        - requires rank :math:`\geq 2`: a bare ``(D,)`` point raises
          ``ValueError``
        - when ``abs(w) > eps`` the remaining components are divided by exactly
          ``w``, with its sign kept (``[[2., 4., -2.]]`` gives ``[[-1., -2.]]``)
        - when ``abs(w) <= eps`` (default ``eps = 1e-8``; the test is a strict
          ``>``) the numerator is instead returned **unchanged**, following
          OpenCV: ``[[2., 4., 0.]]`` gives ``[[2., 4.]]``
        - ``eps`` only picks the branch, it never enters the division, so the
          result carries no bias from it. It is compared against ``w`` in the
          working dtype

    Args:
        points: the points to be transformed of shape ``(*, N, D)``.
        eps: threshold on ``abs(w)`` below which the point is returned
            unchanged instead of divided, to avoid division by zero.

    Returns:
        the points in Euclidean space ``(*, N, D-1)``.

    See Also:
        - :func:`convert_points_to_homogeneous`: the reverse, appending ``w = 1``.
        - :func:`~kornia.geometry.linalg.transform_points`: apply a ``(B, D + 1, D + 1)`` matrix to Euclidean points,
          with this division included.

    Examples:
        >>> input = torch.tensor([[0., 0., 1.]])
        >>> convert_points_from_homogeneous(input)
        tensor([[0., 0.]])

    """
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(points)}")

    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    # we check for points at max_val
    z_vec: torch.Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask: torch.Tensor = torch.abs(z_vec) > eps
    safe_z_vec = torch.where(mask, z_vec, torch.ones_like(z_vec))
    scale = torch.where(mask, 1.0 / safe_z_vec, torch.ones_like(z_vec))

    return scale * points[..., :-1]


def convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    r"""Convert Euclidean points ``(*, N, D)`` to homogeneous ``(*, N, D + 1)`` by appending a 1.

    ``points_h = convert_points_to_homogeneous(points)`` appends the coordinate ``w = 1`` as the **last** component of
    each point ``(*, N, D)``, where :func:`convert_points_from_homogeneous` expects it, and returns ``(*, N, D + 1)`` in
    the input dtype, integer dtypes included.

    Convention:
        - requires rank :math:`\geq 2`: a bare ``(D,)`` point raises
          ``ValueError``

    Args:
        points: the points to be transformed with shape ``(*, N, D)``.

    Returns:
        the points in homogeneous coordinates ``(*, N, D+1)``.

    See Also:
        - :func:`convert_points_from_homogeneous`: the reverse, dividing by ``w`` and dropping it.

    Examples:
        >>> input = torch.tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        tensor([[0., 0., 1.]])

    """
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(points)}")
    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    # Concatenate rather than ``F.pad(points, [0, 1], value=1.0)``: on MPS (seen on torch 2.5.1 and 2.14) a constant
    # pad with a non-zero value returns garbage once the third-from-last dimension reaches 2**16.
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def _convert_affinematrix_to_homography_impl(A: torch.Tensor) -> torch.Tensor:
    if not A.is_floating_point() and not A.is_complex():
        raise ValueError(f"Input A must be a floating point or complex tensor. Got {A.dtype}.")
    H: torch.Tensor = F.pad(A, [0, 0, 0, 1], "constant", value=0.0)
    H[..., -1, -1] += 1.0
    return H


def convert_affinematrix_to_homography(A: torch.Tensor) -> torch.Tensor:
    r"""Extend 2D affine matrices ``(B, 2, 3)`` to ``(B, 3, 3)`` homographies.

    ``H = convert_affinematrix_to_homography(A)`` copies each ``(B, 2, 3)`` affine matrix verbatim into the top of a
    ``(B, 3, 3)`` matrix whose last row is ``[0, 0, 1]``, so that it composes with homographies by matrix product and
    inverts with :func:`torch.linalg.inv`; the input is not modified. For ``(B, 3, 4)`` 3D affine matrices, use
    :func:`convert_affinematrix_to_homography3d`.

    Convention:
        - the rank is enforced to be exactly 3, so batching is **mandatory**: an
          unbatched ``(2, 3)`` matrix raises ``ValueError``

    Note:
        ``A`` must be a floating point or complex tensor; an integer or
        boolean one raises ``ValueError`` naming its dtype.

    Args:
        A: the affine matrix with shape ``(B,2,3)``.

    Returns:
         the homography matrix with shape of ``(B,3,3)``.

    See Also:
        - :func:`convert_affinematrix_to_homography3d`: the 3D form, ``(B, 3, 4)`` to ``(B, 4, 4)``.
        - :func:`~kornia.geometry.transform.invert_affine_transform`: the inverse of a ``(B, 2, 3)`` matrix, returned
          as ``(B, 2, 3)``.

    Examples:
        >>> A = torch.tensor([[[1., 0., 0.],
        ...                    [0., 1., 0.]]])
        >>> convert_affinematrix_to_homography(A)
        tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]])

    """
    if not isinstance(A, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(A)}")

    if not (len(A.shape) == 3 and A.shape[-2:] == (2, 3)):
        raise ValueError(f"Input matrix must be a Bx2x3 tensor. Got {A.shape}")

    return _convert_affinematrix_to_homography_impl(A)


def convert_affinematrix_to_homography3d(A: torch.Tensor) -> torch.Tensor:
    r"""Extend 3D affine matrices ``(B, 3, 4)`` to ``(B, 4, 4)`` homographies.

    ``H = convert_affinematrix_to_homography3d(A)`` copies each ``(B, 3, 4)`` affine matrix verbatim into the top of a
    ``(B, 4, 4)`` matrix whose last row is ``[0, 0, 0, 1]``; the input is not modified. For a rotation and a
    translation given separately, use :func:`Rt_to_matrix4x4`; for ``(B, 2, 3)`` 2D matrices,
    :func:`convert_affinematrix_to_homography`.

    Convention:
        - the rank is enforced to be exactly 3, as in the 2D function: an
          unbatched ``(3, 4)`` matrix raises ``ValueError``

    Note:
        ``A`` must be a floating point or complex tensor; an integer or
        boolean one raises ``ValueError`` naming its dtype.

    Args:
        A: the affine matrix with shape ``(B,3,4)``.

    Returns:
         the homography matrix with shape of ``(B,4,4)``.

    See Also:
        - :func:`convert_affinematrix_to_homography`: the 2D form, ``(B, 2, 3)`` to ``(B, 3, 3)``.
        - :func:`Rt_to_matrix4x4`: the same ``(B, 4, 4)`` matrix from a ``(B, 3, 3)`` rotation and a ``(B, 3, 1)``
          translation.

    Examples:
        >>> A = torch.tensor([[[1., 0., 0., 0.],
        ...                    [0., 1., 0., 0.],
        ...                    [0., 0., 1., 0.]]])
        >>> convert_affinematrix_to_homography3d(A)
        tensor([[[1., 0., 0., 0.],
                 [0., 1., 0., 0.],
                 [0., 0., 1., 0.],
                 [0., 0., 0., 1.]]])

    """
    if not isinstance(A, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(A)}")

    if not (len(A.shape) == 3 and A.shape[-2:] == (3, 4)):
        raise ValueError(f"Input matrix must be a Bx3x4 tensor. Got {A.shape}")

    return _convert_affinematrix_to_homography_impl(A)


def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    r"""Convert axis-angle vectors ``(*, 3)`` in radians to rotation matrices ``(*, 3, 3)``.

    ``R = axis_angle_to_rotation_matrix(axis_angle)`` takes rotation vectors ``(*, 3)``, each the unit axis scaled by
    the angle in radians, and returns the rotation matrices ``(*, 3, 3)`` by Rodrigues' formula, with any number of
    leading dimensions. The matrices act on column vectors from the left and turn by the right-hand rule. For the
    quaternion of the same rotation, use :func:`axis_angle_to_quaternion`; for the reverse,
    :func:`rotation_matrix_to_axis_angle`.

    Convention:
        - any number of leading batch dimensions is accepted: ``(3,)``
          gives ``(3, 3)`` and ``(2, 5, 3)`` gives ``(2, 5, 3, 3)``
        - the input is the rotation axis scaled by the angle, in **radians**:
          ``[[0., 0., pi/2]]`` is a quarter turn about ``+z``,
          while ``[[0., 0., 90.]]`` is 90 *radians* about ``+z`` and returns a
          matrix whose leading entry is ``cos(90) = -0.4481``. The 2-D op
          :func:`~kornia.geometry.conversions.angle_to_rotation_matrix` reads
          **degrees** instead
        - applied on the left to a column vector, ``+theta`` about ``+z`` maps
          ``x_hat`` to ``y_hat`` (right-hand rule): ``[[0., 0., 0.6]]`` sends
          ``(1., 0., 0.)`` to ``(0.8253, 0.5646, 0.)``. The quaternion route
          :func:`~kornia.geometry.conversions.quaternion_to_rotation_matrix`
          turns the same way
        - ``angle_axis_to_rotation_matrix`` is the deprecated alias of this
          function since 0.7.0; it emits a ``DeprecationWarning`` and forwards
          to this function, returning an equal result
        - differentiable at the identity rotation. The ``sqrt`` whose
          derivative is unbounded there never sees its zero radicand, so the
          gradient is the analytic limit of the surrounding map rather than
          ``nan``. The guard is elementwise, and away from the identity it
          moves no forward bit

    Args:
        axis_angle: tensor of 3d vector of axis-angle rotations in radians with shape ``(*, 3)``.

    Returns:
        tensor of rotation matrices of shape ``(*, 3, 3)``.

    See Also:
        - :func:`rotation_matrix_to_axis_angle`: the reverse conversion, matrix to axis-angle.
        - :func:`axis_angle_to_quaternion`: the ``(w, x, y, z)`` quaternion of the same rotation.
        - :func:`quaternion_to_rotation_matrix`: the same matrix from a quaternion.
        - :func:`angle_to_rotation_matrix`: a 2D ``(*, 2, 2)`` rotation from an angle in degrees.

    Example:
        >>> input = torch.tensor([[0., 0., 0.]])
        >>> axis_angle_to_rotation_matrix(input)  # doctest: +ELLIPSIS
        tensor([[[1., ...0., 0.],
                 [0., 1., ...0.],
                 [...0., 0., 1.]]])

        >>> input = torch.tensor([[1.5708, 0., 0.]])
        >>> axis_angle_to_rotation_matrix(input)
        tensor([[[ 1.0000e+00,  0.0000e+00,  0.0000e+00],
                 [ 0.0000e+00, -3.6200e-06, -1.0000e+00],
                 [ 0.0000e+00,  1.0000e+00, -3.6200e-06]]])

    """
    if not isinstance(axis_angle, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(axis_angle)}")

    if not axis_angle.shape[-1] == 3:
        raise ValueError(f"Input size must be a (*, 3) tensor. Got {axis_angle.shape}")

    def _compute_rotation_matrix(axis_angle: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        theta = torch.sqrt(theta2)
        wxyz = axis_angle / theta.unsqueeze(-1)  # (*, 3)
        wx, wy, wz = wxyz.unbind(dim=-1)  # (*,)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        one_minus_cos = 1.0 - cos_theta

        wxwy = wx * wy
        wxwz = wx * wz
        wywz = wy * wz

        r00 = cos_theta + wx * wx * one_minus_cos
        r01 = wxwy * one_minus_cos - wz * sin_theta
        r02 = wy * sin_theta + wxwz * one_minus_cos

        r10 = wz * sin_theta + wxwy * one_minus_cos
        r11 = cos_theta + wy * wy * one_minus_cos
        r12 = -wx * sin_theta + wywz * one_minus_cos

        r20 = -wy * sin_theta + wxwz * one_minus_cos
        r21 = wx * sin_theta + wywz * one_minus_cos
        r22 = cos_theta + wz * wz * one_minus_cos

        return torch.stack(
            [
                torch.stack([r00, r01, r02], dim=-1),
                torch.stack([r10, r11, r12], dim=-1),
                torch.stack([r20, r21, r22], dim=-1),
            ],
            dim=-2,
        )

    def _compute_rotation_matrix_taylor(axis_angle: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        rx, ry, rz = axis_angle.unbind(-1)
        k_one = torch.ones_like(rx)

        # Rodrigues' formula R = I + a [v]x + b [v]x^2, with the series of its
        # coefficients a = sin(theta) / theta and b = (1 - cos(theta)) / theta^2
        # kept to the theta^2 term. Stopping at a = 1 and b = 1/2 left an error of
        # theta^3 / 6 per entry, 1.3e-10 at theta = 1e-3 in float64. The
        # first dropped terms are theta^4 / 120 and theta^4 / 720, which puts the
        # error at theta^5 / 120, below float64 rounding across the branch.
        a = k_one - theta2 / 6.0
        b = 0.5 * k_one - theta2 / 24.0

        rx2, ry2, rz2 = rx * rx, ry * ry, rz * rz
        rxry, rxrz, ryrz = rx * ry, rx * rz, ry * rz

        return torch.stack(
            [
                k_one - b * (ry2 + rz2),
                -a * rz + b * rxry,
                a * ry + b * rxrz,
                a * rz + b * rxry,
                k_one - b * (rx2 + rz2),
                -a * rx + b * ryrz,
                -a * ry + b * rxrz,
                a * rx + b * ryrz,
                k_one - b * (rx2 + ry2),
            ],
            dim=-1,
        ).reshape(list(axis_angle.shape[:-1]) + [3, 3])

    theta2 = (axis_angle * axis_angle).sum(dim=-1)
    mask = theta2 > 1e-6

    # Rows on the Taylor branch feed the discarded Rodrigues branch a stand-in theta2 of 1, so its backward never
    # differentiates sqrt at 0. A clamp floor is no guard here: 1e-12 underflows to 0 in float16.
    safe_theta2 = torch.where(mask, theta2, torch.ones_like(theta2))
    rot_normal = _compute_rotation_matrix(axis_angle, safe_theta2)  # (*,3,3)
    rot_taylor = _compute_rotation_matrix_taylor(axis_angle, theta2)  # (*,3,3)

    return torch.where(mask[..., None, None], rot_normal, rot_taylor)


@deprecated(replace_with="axis_angle_to_rotation_matrix", version="0.7.0")
def angle_axis_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Deprecated: use :func:`~kornia.geometry.conversions.axis_angle_to_rotation_matrix`.

    Each call emits a ``DeprecationWarning`` (deprecated since 0.7.0) and returns what the new name returns.
    """
    return axis_angle_to_rotation_matrix(axis_angle)


def rotation_matrix_to_axis_angle(rotation_matrix: torch.Tensor, check_rotation: bool = False) -> torch.Tensor:
    r"""Convert rotation matrices ``(*, 3, 3)`` to axis-angle vectors ``(*, 3)`` in radians.

    ``axis_angle = rotation_matrix_to_axis_angle(R)`` returns, for each rotation matrix ``(*, 3, 3)``, the Rodrigues
    vector ``(*, 3)``: the unit axis scaled by the angle in radians, at most ``pi``, the form
    :func:`axis_angle_to_rotation_matrix` takes back. It goes through :func:`rotation_matrix_to_quaternion` and
    :func:`quaternion_to_axis_angle`. The input is not checked to be a rotation unless ``check_rotation=True``. For the
    ``(w, x, y, z)`` quaternion instead, use :func:`rotation_matrix_to_quaternion`.

    Convention:
        - any number of leading batch dimensions is accepted: ``(3, 3)``
          gives ``(3,)`` and ``(2, 5, 3, 3)`` gives ``(2, 5, 3)``
        - the output is the parametrization
          :func:`~kornia.geometry.conversions.axis_angle_to_rotation_matrix`
          consumes, at every rank: the ``(3,)`` and ``(2, 5, 3)``
          results above can be fed straight back
        - ``rotation_matrix_to_axis_angle(axis_angle_to_rotation_matrix(v))``
          returns ``v`` to roundoff for angles below ``pi``, the low-angle
          Taylor branch (``theta <= 1e-3``) included. A longer ``v`` comes
          back as the equivalent vector of angle at most ``pi``:
          ``[0., 0., 4.]`` returns ``[0., 0., -2.2832]``, a turn by
          ``2 * pi - 4`` about ``-z``
        - by default the input is **not** checked for being a rotation matrix:
          ``zeros(3, 3)`` returns ``[0., 0., 3.1416]`` and ``2 * eye(3)``
          returns ``[0., 0., 0.]``. Pass ``check_rotation=True`` to reject
          such an input instead
        - ``rotation_matrix_to_angle_axis`` is the deprecated alias of this
          function since 0.7.0

    .. warning::
        By default a reflection (``det = -1``) is returned as a rotation:
        ``diag(-1, 1, 1)`` gives ``[0., 0., 0.]``, the identity. Pass
        ``check_rotation=True`` to raise a ``ValueError`` instead.

    Args:
        rotation_matrix: rotation matrix of shape ``(*, 3, 3)``.
        check_rotation: if ``True``, raise ``ValueError`` unless every input
            is a rotation: ``max|R @ R^T - I|`` within 100 eps of the dtype of
            ``rotation_matrix`` (16 eps for float16 and bfloat16) and
            ``det(R) > 0``. Defaults to ``False`` (unchecked).

    Returns:
        Rodrigues vector transformation of shape ``(*, 3)``.

    See Also:
        - :func:`axis_angle_to_rotation_matrix`: the reverse conversion, axis-angle to matrix.
        - :func:`rotation_matrix_to_quaternion`: the ``(w, x, y, z)`` quaternion of the same matrix.
        - :func:`quaternion_to_axis_angle`: the same vector from a quaternion.

    Example:
        >>> input = torch.tensor([[1., 0., 0.],
        ...                       [0., 1., 0.],
        ...                       [0., 0., 1.]])
        >>> rotation_matrix_to_axis_angle(input)
        tensor([0., 0., 0.])

        >>> input = torch.tensor([[1., 0., 0.],
        ...                       [0., 0., -1.],
        ...                       [0., 1., 0.]])
        >>> rotation_matrix_to_axis_angle(input)
        tensor([1.5708, 0.0000, 0.0000])

    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    if check_rotation:
        _check_is_rotation(rotation_matrix, "rotation_matrix_to_axis_angle")

    quaternion: torch.Tensor = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_axis_angle(quaternion)


@deprecated(replace_with="rotation_matrix_to_axis_angle", version="0.7.0")
def rotation_matrix_to_angle_axis(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Deprecated: use :func:`~kornia.geometry.conversions.rotation_matrix_to_axis_angle`.

    Each call emits a ``DeprecationWarning`` (deprecated since 0.7.0) and returns what the new name returns with its
    default ``check_rotation=False``: this name takes no ``check_rotation`` argument, so its input is never checked.
    """
    return rotation_matrix_to_axis_angle(rotation_matrix)


def rotation_matrix_to_quaternion(
    rotation_matrix: torch.Tensor, eps: float = 1.0e-8, check_rotation: bool = False
) -> torch.Tensor:
    r"""Convert rotation matrices ``(*, 3, 3)`` to quaternions ``(*, 4)`` in ``(w, x, y, z)`` order.

    ``q = rotation_matrix_to_quaternion(R)`` returns, for each rotation matrix ``(*, 3, 3)``, the unit quaternion
    ``(*, 4)`` with the real part first, with any number of leading dimensions. Of the two quaternions ``q`` and ``-q``
    of a rotation it returns either, not always the one with ``w >= 0``. The input is not checked to be a rotation
    unless ``check_rotation=True``. For the reverse, use :func:`quaternion_to_rotation_matrix`; for the axis-angle
    vector, :func:`rotation_matrix_to_axis_angle`.

    Convention:
        - the ``(w, x, y, z)`` layout, and its silent misreading, are spelled
          out on :func:`~kornia.geometry.conversions.quaternion_to_rotation_matrix`
        - the sign is **not** canonicalised to ``w >= 0``. For
          ``trace(R) > 0`` the returned ``w`` is non-negative, but for
          ``trace(R) <= 0`` it is the dominant of ``x, y, z`` that is forced
          non-negative and ``w`` may be negative: 170 degrees about
          ``(1, 2, -3)/sqrt(14)`` (``trace = -0.9696``) returns
          ``[-0.0872, -0.2662, -0.5325, 0.7987]``
        - by default the input is **not** checked for being a rotation matrix;
          see the degenerate inputs listed on
          :func:`~kornia.geometry.conversions.rotation_matrix_to_axis_angle`.
          Pass ``check_rotation=True`` to reject such an input instead

    Args:
        rotation_matrix: the rotation matrix to convert with shape ``(*, 3, 3)``.
        eps: floor on each square-root radicand. It is never added to the
            radicand: a radicand above ``eps`` is used as is, and one at or
            below it (for example the negative radicand of a discarded branch
            on a slightly non-orthogonal input) gives ``2 * sqrt(eps)`` with a
            zero gradient. For an exact rotation matrix the selected radicand is
            at least ``1``, so ``eps`` does not change the result. The divisions
            are separately guarded against a zero denominator.
        check_rotation: if ``True``, raise ``ValueError`` unless every input
            is a rotation: ``max|R @ R^T - I|`` within 100 eps of the dtype of
            ``rotation_matrix`` (16 eps for float16 and bfloat16) and
            ``det(R) > 0``. Defaults to ``False`` (unchecked).

    Return:
        the rotation in quaternion with shape ``(*, 4)``.

    See Also:
        - :func:`quaternion_to_rotation_matrix`: the reverse conversion, quaternion to matrix.
        - :func:`rotation_matrix_to_axis_angle`: the axis-angle vector of the same matrix.
        - :func:`axis_angle_to_quaternion`: the quaternion of an axis-angle vector instead.
        - :meth:`~kornia.geometry.quaternion.Quaternion.from_matrix`: the same quaternion as a
          :class:`~kornia.geometry.quaternion.Quaternion` object.

    Example:
        >>> input = torch.tensor([[1., 0., 0.],
        ...                       [0., 1., 0.],
        ...                       [0., 0., 1.]])
        >>> rotation_matrix_to_quaternion(input)
        tensor([1., 0., 0., 0.])

    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    if check_rotation:
        _check_is_rotation(rotation_matrix, "rotation_matrix_to_quaternion")

    def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec: torch.Tensor = rotation_matrix.reshape(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    # eps floors the radicand instead of being added to it, so it never shifts a valid result.
    radicand_floor: float = max(eps, 0.0)

    def _safe_sqrt_sq(r: torch.Tensor) -> torch.Tensor:
        # 2 * sqrt(max(r, eps)) with a finite backward everywhere: sqrt is only ever differentiated at a
        # radicand above the floor (or at a substituted 1), and a radicand at or below the floor takes the
        # constant 2 * sqrt(eps) from the other where arm. No clamp: its derivative at the bound depends on the
        # torch version. A NaN radicand is passed through, so an invalid matrix never becomes a finite quaternion.
        mask = r > radicand_floor
        safe_r = torch.where(mask, r, torch.ones_like(r))
        out = torch.where(mask, torch.sqrt(safe_r) * 2.0, torch.full_like(r, 2.0 * radicand_floor**0.5))
        return torch.where(torch.isnan(r), r, out)

    def trace_positive_cond() -> torch.Tensor:
        sq = _safe_sqrt_sq(trace + 1.0)  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1() -> torch.Tensor:
        sq = _safe_sqrt_sq(1.0 + m00 - m11 - m22)  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2() -> torch.Tensor:
        sq = _safe_sqrt_sq(1.0 + m11 - m00 - m22)  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3() -> torch.Tensor:
        sq = _safe_sqrt_sq(1.0 + m22 - m00 - m11)  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: torch.Tensor = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    r"""Scale quaternions ``(*, 4)`` to unit length, keeping their sign and component order.

    ``q_unit = normalize_quaternion(q)`` divides each quaternion ``(*, 4)`` by its L2 norm and returns ``(*, 4)``. It
    reads no component order, so the quaternion can be in ``(x, y, z, w)`` or ``(w, x, y, z)`` format. The norm is
    floored at ``eps``, so with a positive ``eps`` the zero quaternion returns zeros rather than ``nan``. The
    conversions of this module that take a quaternion accept a non-unit one, so they need no call here first.

    Convention:
        - it is an L2 normalisation of the **last** axis and nothing else, which
          is why the ``(x, y, z, w)`` or ``(w, x, y, z)`` phrasing above is
          accurate here: ``[4., 3., 2., 1.]`` normalises to the reversal of what
          ``[1., 2., 3., 4.]`` gives
        - the sign is preserved — ``normalize_quaternion(-q)`` is exactly
          ``-normalize_quaternion(q)``, there is no ``w >= 0``
          canonicalisation
        - the divisor is the norm floored at ``eps``, so a quaternion with
          ``0 < ||q|| < eps`` returns ``q / eps``, which is **not** a unit
          quaternion, and no error is raised: with the default ``eps = 1e-12``,
          ``[1e-13, 0., 0., 0.]`` returns ``[0.1, 0., 0., 0.]``. No realistic
          rotation has such a norm; the case that occurs in practice, the zero
          quaternion, returns ``zeros(4)``. With ``eps=0.0`` the zero quaternion
          returns ``nan``. In ``float16`` a positive ``eps`` is floored at the
          smallest subnormal.

    Args:
        quaternion: a tensor containing a quaternion to be normalized.
          The tensor can be of shape ``(*, 4)``.
        eps: floor on the norm used as the divisor; see the Convention above.

    Return:
        the normalized quaternion of shape ``(*, 4)``.

    See Also:
        - :func:`quaternion_to_rotation_matrix`: normalizes with this function before building the matrix.
        - :meth:`~kornia.geometry.quaternion.Quaternion.normalize`: the same normalization of a
          :class:`~kornia.geometry.quaternion.Quaternion` object.

    Example:
        >>> quaternion = torch.tensor((1., 0., 1., 0.))
        >>> normalize_quaternion(quaternion)
        tensor([0.7071, 0.0000, 0.7071, 0.0000])

    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")

    return _normalize_last_dim(quaternion, eps)


def _normalize_last_dim(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Divide ``x`` by the L2 norm of its last axis, floored at ``eps``.

    The arithmetic of :func:`normalize_quaternion`, shared with :meth:`kornia.geometry.vector.Vector3.normalized`,
    :meth:`kornia.geometry.vector.Vector2.normalized` and :meth:`kornia.geometry.line.ParametrizedLine.through`.
    Its values match ``torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)`` wherever the norm is nonzero. Their
    default ``eps=1e-12`` rounds to ``0`` in float16, where ``normalize`` turns a zero vector into ``0 / 0 = NaN``
    (#4021, #5062); here a row whose norm is exactly zero takes a separate arm (``x / eps``, or a constant zero where
    float16 cannot hold ``1 / eps``), so with a positive ``eps`` the zero vector stays zero in every dtype.
    """
    safe_eps: float = max(eps, 5.960464477539063e-08) if x.dtype == torch.float16 and eps > 0.0 else eps
    norm = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
    # Only an exactly-zero norm takes the constant arms below; a NaN norm compares unequal to zero and goes
    # through the division, so NaN in gives NaN out. The eps floor is a value floor selected by torch.where,
    # so its derivative does not depend on the torch version the way clamp's derivative at the bound does.
    mask = norm != 0.0
    safe_norm = torch.where(mask, norm, torch.ones_like(norm))
    denom = torch.where(safe_norm < safe_eps, torch.full_like(safe_norm, safe_eps), safe_norm)
    out = x / denom
    if eps == 0.0:
        return torch.where(mask, out, torch.full_like(x, float("nan")))
    if x.dtype == torch.float16 and safe_eps * 65504.0 < 1.0:
        # The exact derivative of x / eps at x = 0 is I / eps, which overflows float16 (#4623); use a zero gradient.
        return torch.where(mask, out, torch.zeros_like(x))
    # Below eps the function is x / eps, so the zero vector keeps its exact derivative I / eps.
    return torch.where(mask, out, x / torch.full_like(x, safe_eps))


# based on:
# https://github.com/matthew-brett/transforms3d/blob/8965c48401d9e8e66b6a8c37c65f2fc200a076fa/transforms3d/quaternions.py#L101
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py#L247


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    r"""Convert ``(w, x, y, z)`` quaternions ``(*, 4)`` to rotation matrices ``(*, 3, 3)``.

    ``R = quaternion_to_rotation_matrix(q)`` takes quaternions ``(*, 4)`` with the **real part first**, in the Hamilton
    convention, and returns the rotation matrices ``(*, 3, 3)``, which act on column vectors from the left. The input
    is normalized first, so it need not be unit, and ``q`` and ``-q`` give the same matrix. A quaternion in
    ``(x, y, z, w)`` order is not detected and returns a valid-looking wrong rotation. For the reverse, use
    :func:`rotation_matrix_to_quaternion`; from an axis-angle vector, :func:`axis_angle_to_rotation_matrix`.

    Convention:
        - ``[1., 0., 0., 0.]`` gives the identity and ``[0., 0., 0., 1.]`` gives
          ``[[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]``, a half turn about
          ``+z`` — so a caller passing ``(x, y, z, w)`` gets a valid-looking
          rotation matrix and no error
        - ``q`` and ``-q`` are the same rotation (double cover) and return
          bit-identical matrices
        - the input is normalised internally with
          :func:`~kornia.geometry.conversions.normalize_quaternion`, so
          ``quaternion_to_rotation_matrix(2 * q)`` is bitwise
          ``quaternion_to_rotation_matrix(q)``; other rescalings move the
          matrix by the working dtype's rounding, as long as the norm of the
          rescaled quaternion neither overflows nor falls below the
          normalisation floor ``eps = 1e-12`` (see that function's Convention)
        - :func:`~kornia.geometry.conversions.quaternion_to_axis_angle`,
          :func:`~kornia.geometry.conversions.quaternion_exp_to_log`, and
          :func:`~kornia.geometry.conversions.euler_from_quaternion` are
          scale-safe too
        - applied on the left to a column vector, ``+theta`` about ``+z`` maps
          ``x_hat`` to ``y_hat`` (right-hand rule)
        - the output dtype follows the input at every shape
        - the zero quaternion, which is not a rotation, returns the identity
          matrix rather than raising: it stays zero through the normalisation,
          and the matrix formula gives the identity for it

    Args:
        quaternion: a tensor containing a quaternion to be converted.
          The tensor can be of shape ``(*, 4)``.

    Return:
        the rotation matrix of shape ``(*, 3, 3)``.

    See Also:
        - :func:`rotation_matrix_to_quaternion`: the reverse conversion, matrix to quaternion.
        - :func:`quaternion_to_axis_angle`: the axis-angle vector of the same quaternion.
        - :func:`axis_angle_to_rotation_matrix`: the same matrix from an axis-angle vector.
        - :func:`normalize_quaternion`: the normalization applied to the input first.
        - :meth:`~kornia.geometry.quaternion.Quaternion.matrix`: the same matrix of a
          :class:`~kornia.geometry.quaternion.Quaternion` object.

    Example:
        >>> quaternion = torch.tensor((0., 0., 0., 1.))
        >>> quaternion_to_rotation_matrix(quaternion)
        tensor([[-1.,  0.,  0.],
                [ 0., -1.,  0.],
                [ 0.,  0.,  1.]])

    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")

    # normalize the input quaternion
    quaternion_norm: torch.Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    w = quaternion_norm[..., 0]
    x = quaternion_norm[..., 1]
    y = quaternion_norm[..., 2]
    z = quaternion_norm[..., 3]

    # compute the actual conversion
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z
    one: float = 1.0

    matrix_flat: torch.Tensor = torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    )

    # this slightly awkward construction of the output shape is to satisfy torchscript
    output_shape = [*list(quaternion.shape[:-1]), 3, 3]
    return matrix_flat.reshape(output_shape)


def quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert ``(w, x, y, z)`` quaternions ``(*, 4)`` to axis-angle vectors ``(*, 3)`` in radians.

    ``axis_angle = quaternion_to_axis_angle(q)`` returns, for each quaternion ``(*, 4)`` with the real part first, the
    rotation vector ``(*, 3)``: the unit axis scaled by the angle in radians, at most ``pi``. The input need not be
    unit, and ``q`` and ``-q`` give the same vector unless ``w = 0``. For the reverse, use
    :func:`axis_angle_to_quaternion`; for the logarithm map, which is half this vector for ``w >= 0`` and tells ``q``
    from ``-q``, :func:`quaternion_exp_to_log`.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Convention:
        - the ``(w, x, y, z)`` layout is spelled out on
          :func:`~kornia.geometry.conversions.quaternion_to_rotation_matrix`
        - for ``w != 0`` the double cover is collapsed: ``q`` and ``-q`` return
          the same vector, the representative with ``|theta| <= pi``
        - at exactly ``w = 0`` — a half turn, and ``0.`` is exactly
          representable — the collapse does not happen and the two return exact
          negations: ``[0., 1., 0., 0.]`` gives ``[pi, 0., 0.]`` while
          ``[-0., -1., 0., 0.]`` gives ``[-pi, 0., 0.]``, the same rotation
          written the other way round
        - the input need not be unit: scaling ``q`` by a power of two returns a
          bit-identical vector, and other scales move it by the working
          dtype's rounding, as long as the squares of the components neither
          overflow nor underflow
        - ``quaternion_to_angle_axis`` is the deprecated alias of this function
          since 0.7.0
        - differentiable at the identity quaternion. The ``sqrt`` whose
          derivative is unbounded there never sees its zero radicand, so the
          gradient is the analytic limit of the surrounding map rather than
          ``nan``. The guard is elementwise, and away from the identity it
          moves no forward bit

    Args:
        quaternion: tensor with quaternions.

    Return:
        tensor with axis angle of rotation.

    Shape:
        - Input: ``(*, 4)`` where `*` means, any number of dimensions
        - Output: ``(*, 3)``

    See Also:
        - :func:`axis_angle_to_quaternion`: the reverse conversion, axis-angle to quaternion.
        - :func:`quaternion_to_rotation_matrix`: the rotation matrix of the same quaternion.
        - :func:`quaternion_exp_to_log`: the logarithm map, half this vector for ``w >= 0``; it tells ``q`` from
          ``-q``.
        - :func:`rotation_matrix_to_axis_angle`: the same vector from a rotation matrix.

    Example:
        >>> quaternion = torch.tensor((1., 0., 0., 0.))
        >>> quaternion_to_axis_angle(quaternion)
        tensor([0., 0., 0.])

    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape Nx4 or 4. Got {quaternion.shape}")

    # unpack input and compute conversion
    cos_theta = quaternion[..., 0]
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]

    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    # the sqrt and the division below are only taken where the radicand is non zero. the inner
    # `where` keeps the identity rotation away from the sqrt so its backward pass stays finite.
    pos: torch.Tensor = sin_squared_theta > 0.0
    safe_sin_squared_theta: torch.Tensor = torch.where(pos, sin_squared_theta, torch.ones_like(sin_squared_theta))
    sin_theta: torch.Tensor = torch.where(pos, torch.sqrt(safe_sin_squared_theta), torch.zeros_like(sin_squared_theta))
    # `two_theta` only reaches the output through `k_pos`, which `k`'s final `where` selects on
    # `pos` -- so every element with a zero vector part gets an exact 0.0 gradient here. That is
    # not enough to make the backward safe: `atan2`'s backward multiplies by
    # `1 / (sin_theta**2 + cos_theta**2)`, and where that reciprocal is `inf` the masked 0.0
    # meets it as `0 * inf` -> `nan`, which propagates into the caller's gradient. The reciprocal
    # is `inf` for the exact-zero quaternion (0/0, `nan` on torch <= 2.9.1 and 0 on 2.14) and
    # also whenever `w**2` underflows -- `w = 1e-30` in float32, `|w| < 2.4e-4` in float16. So
    # substitute a 1 into `atan2` across the whole masked branch, not just at `w == 0`; don't
    # simplify the shield away. The substitution goes through `torch.where`, which hands `atan2`
    # a contiguous tensor instead of the stride-4 view of the input and can therefore select a
    # different kernel and move the forward by one ulp -- so the value comes from the unshielded
    # expression and only the gradient flows through the shielded one.
    safe_cos_for_atan2: torch.Tensor = torch.where(pos, cos_theta, torch.ones_like(cos_theta))
    two_theta_shielded: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -safe_cos_for_atan2), torch.atan2(sin_theta, safe_cos_for_atan2)
    )
    two_theta_value: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta)
    )
    two_theta: torch.Tensor = two_theta_shielded + (two_theta_value.detach() - two_theta_shielded.detach())

    k_pos: torch.Tensor = two_theta / torch.where(pos, sin_theta, torch.ones_like(sin_theta))
    # The zero-vector-part branch's analytic limit is 2/w (w = cos_theta), not the constant 2.0
    # that implicitly assumed w=1 -- see #4237. That constant gave the wrong sign at the negative
    # unit identity (-1,0,0,0), which is the same rotation as (1,0,0,0), and the wrong magnitude
    # at any non-unit "identity" (e.g. w=2), which this function explicitly allows. The division
    # is gated on the branch that actually selects it, `~pos`, and not merely on `w != 0`: for a
    # rotation near a half turn `w` is small but non-zero, `(1/w)**2` overflows in the reciprocal
    # backward, and the 0.0 that this masked branch receives would meet that `inf` as `0 * inf`.
    # In float16 that is every rotation within ~0.45 degrees of 180. The exact-zero quaternion,
    # which is not a valid rotation, is left at 0 with a zero gradient; it previously had 2 in
    # each vector slot and, on torch <= 2.9.1, `nan` in w's.
    #
    # `safe_cos_theta` is detached because `k_neg` is only ever multiplied by a vector component
    # that this branch has already established is zero, so `d(out)/dw = q_i * -2/w**2` is exactly
    # zero here -- but computing it that way evaluates `-2/w**2`, which overflows to `inf` for
    # small `w` and turns the exact zero into `0 * inf` -> `nan` (`w = 1e-30` in float32, and
    # `|w| < 0.0055` in float16, both of which already produce a `nan` here on main). Detaching
    # returns that exact zero without the intermediate overflow. The only term it discards is
    # from the corner where a vector component is non-zero but its square underflows to zero, in
    # which case the discarded value is of order `|v| / w**2` in a regime the forward has already
    # flushed. The `2 / w` coefficient itself, which is what #4237 is about, is untouched.
    neg_branch: torch.Tensor = ~pos & (cos_theta != 0.0)
    safe_cos_theta: torch.Tensor = torch.where(neg_branch, cos_theta, torch.ones_like(cos_theta))
    k_neg: torch.Tensor = torch.where(neg_branch, 2.0 / safe_cos_theta.detach(), torch.zeros_like(cos_theta))
    k: torch.Tensor = torch.where(pos, k_pos, k_neg)

    axis_angle: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    axis_angle[..., 0] += q1 * k
    axis_angle[..., 1] += q2 * k
    axis_angle[..., 2] += q3 * k
    return axis_angle


@deprecated(replace_with="quaternion_to_axis_angle", version="0.7.0")
def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Deprecated: use :func:`~kornia.geometry.conversions.quaternion_to_axis_angle`.

    Each call emits a ``DeprecationWarning`` (deprecated since 0.7.0) and returns what the new name returns.
    """
    return quaternion_to_axis_angle(quaternion)


def quaternion_log_to_exp(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    r"""Map log 3-vectors ``(*, 3)`` to ``(w, x, y, z)`` quaternions ``(*, 4)``, the exponential map.

    ``q = quaternion_log_to_exp(v)`` takes 3-vectors ``(*, 3)``, the logarithms of unit quaternions, and returns the
    quaternions ``(cos(||v||), sin(||v||) v / ||v||)`` of shape ``(*, 4)``, real part first. Despite its name, the
    argument ``quaternion`` is the 3-vector, which is **half** the axis-angle vector of the rotation; for the
    quaternion of an axis-angle vector, use :func:`axis_angle_to_quaternion`. For the reverse, use
    :func:`quaternion_exp_to_log`.

    Convention:
        - the ``(w, x, y, z)`` layout is spelled out on
          :func:`~kornia.geometry.conversions.quaternion_to_rotation_matrix`
        - the map is ``w = cos(||v||)`` with vector part
          ``sin(||v||) * v / ||v||``, and every finite ``v`` returns a unit
          quaternion up to the working dtype's rounding: ``||v||`` is computed
          as ``s * ||v / s||`` with ``s = max |v_i|``, so it cannot overflow,
          and ``float16``/``bfloat16`` are computed in ``float32``, so the zero
          vector returns ``[1., 0., 0., 0.]`` at every dtype
        - ``pi/2 < ||v|| < 3 * pi/2`` lands in the ``w < 0`` half of the double
          cover — at ``||v|| = 2`` the real part is ``-0.4161468365471424``.
          ``w`` is ``cos(||v||)``, so the sign turns over again beyond that
          interval rather than staying negative
        - ``v`` is therefore **half** the axis-angle vector: the result agrees
          with ``axis_angle_to_quaternion(2 * v)`` to the rounding of the
          working dtype, but **not** bit-for-bit: ``[0.15, 0.2, 0.25]`` already
          disagrees at ``bfloat16``, and in ``float64`` most random vectors do
          too
        - for a **unit** ``q``, ``quaternion_log_to_exp(quaternion_exp_to_log(q))``
          returns ``q`` up to rounding, except that the pure-real
          ``[-1., 0., 0., 0.]`` comes back as ``[1., 0., 0., 0.]`` — the other
          half of the same rotation

    Args:
        quaternion: the log quaternion, a tensor of shape ``(*, 3)``.
        eps: a small number for clamping.

    Return:
        the quaternion exponential map of shape ``(*, 4)``.

    See Also:
        - :func:`quaternion_exp_to_log`: the reverse, the logarithm map.
        - :func:`axis_angle_to_quaternion`: the same quaternion from the full axis-angle vector ``2 * v``.

    Example:
        >>> quaternion = torch.tensor((0., 0., 0.))
        >>> quaternion_log_to_exp(quaternion, eps=torch.finfo(quaternion.dtype).eps)
        tensor([1., 0., 0., 0.])

    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 3:
        raise ValueError(f"Input must be a tensor of shape (*, 3). Got {quaternion.shape}")

    # compute the quaternion norm in a way that cannot overflow (kornia#3975):
    #  - float16/bfloat16 inputs are upcast to float32, where the default eps = 1e-8 is
    #    representable, so the clamp below is not a no-op and the float16 zero vector stays the
    #    identity quaternion (the kornia#3966 underflow class does not bite on this function);
    #  - the norm is computed as scale * ||v / scale|| with scale = max |v_i|, so the squared
    #    terms are bounded by the number of components and the whole finite input range of every
    #    dtype returns a finite unit quaternion.
    orig_dtype = quaternion.dtype
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    work = quaternion.to(work_dtype)

    scale: torch.Tensor = work.abs().max(dim=-1, keepdim=True).values
    norm_q: torch.Tensor = scale * torch.norm(work / scale.clamp(min=eps), p=2, dim=-1, keepdim=True)
    norm_q = norm_q.clamp(min=eps)

    # compute scalar and vector
    quaternion_vector: torch.Tensor = work * torch.sin(norm_q) / norm_q
    quaternion_scalar: torch.Tensor = torch.cos(norm_q)

    # compose quaternion and return
    return torch.cat((quaternion_scalar, quaternion_vector), dim=-1).to(orig_dtype)


def quaternion_exp_to_log(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    r"""Map ``(w, x, y, z)`` quaternions ``(*, 4)`` to log 3-vectors ``(*, 3)``, the logarithm map.

    ``v = quaternion_exp_to_log(q)`` takes quaternions ``(*, 4)`` with the real part first and returns the 3-vectors
    ``atan2(||u||, w) u / max(||u||, eps)`` of shape ``(*, 3)``, with ``u`` the vector part. While ``||u||`` is at
    least ``eps``, the input need not be unit and :func:`quaternion_log_to_exp` maps the result back to the unit
    quaternion of ``q``; below ``eps`` the log shrinks by ``||u|| / eps``, so ``(-1, 0, 0, 0)`` comes back as
    ``(1, 0, 0, 0)``. For ``w >= 0`` the result is half the
    axis-angle vector; for ``w < 0`` it is not, because this map tells ``q`` from ``-q``. For the axis-angle vector,
    use :func:`quaternion_to_axis_angle`.

    Convention:
        - for a **unit** ``q`` on the ``w >= 0`` half of the double cover the
          result is ``quaternion_to_axis_angle(q) / 2``. On the ``w < 0`` half the two
          part company: this function keeps the sign of ``w``, with the angle
          ``atan2(||u||, w)`` in ``[0, pi]`` for the vector part ``u``, while
          :func:`~kornia.geometry.conversions.quaternion_to_axis_angle`
          collapses the double cover. For the ``q`` whose log is ``v``,
          ``quaternion_exp_to_log(-q)`` is ``-(pi - ||v||) * v / ||v||``
        - ``quaternion_exp_to_log(quaternion_log_to_exp(v))`` returns ``v`` for
          ``0 < ||v|| < pi``. A longer ``v`` comes back shifted along its own
          direction by a multiple of ``2 * pi`` to a length of at most ``pi``:
          ``pi + 0.5`` and ``3 * pi + 0.5`` both come back as
          ``-2.6415926535897936``, ``2 * pi + 0.5`` as ``0.5``. At exactly
          ``||v|| = pi`` in ``float64`` it collapses to nearly ``0``: ``cos(pi)``
          rounds to ``-1`` and the vector part falls under the ``eps`` clamp.
          Both are properties of the map, not defects
        - ``float16``/``bfloat16`` are computed in ``float32``, so the identity
          returns ``[0., 0., 0.]`` at every dtype, and the gradient is finite
          at ``w = +-1``, the identity included
        - the map is invariant to a positive scale while ``||u||`` stays at least ``eps``: a non-unit quaternion
          produces the same log as its normalisation, and ``-q`` another one

    Args:
        quaternion: a tensor containing a quaternion to be converted.
          The tensor can be of shape ``(*, 4)``.
        eps: a small number for clamping.

    Return:
        the quaternion log map of shape ``(*, 3)``.

    See Also:
        - :func:`quaternion_log_to_exp`: the reverse, the exponential map.
        - :func:`quaternion_to_axis_angle`: the axis-angle vector, twice this one for ``w >= 0``.

    Example:
        >>> quaternion = torch.tensor((1., 0., 0., 0.))
        >>> quaternion_exp_to_log(quaternion, eps=torch.finfo(quaternion.dtype).eps)
        tensor([0., 0., 0.])

    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")

    # use float32 for float16/bfloat16 so the default eps is representable (kornia#3966)
    orig_dtype = quaternion.dtype
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    work = quaternion.to(work_dtype)
    quaternion_scalar = work[..., 0:1]
    quaternion_vector = work[..., 1:4]

    norm_v: torch.Tensor = torch.norm(quaternion_vector, p=2, dim=-1, keepdim=True)
    norm_q: torch.Tensor = norm_v.clamp(min=eps)

    # atan2(||v||, w) is scale-invariant, needs no clamp, and has a bounded derivative at w = +-1.
    # Its derivative is 0/0 at the zero quaternion, where torch 2.5.1 returns nan and 2.14 returns 0:
    # differentiate there at w = 1 instead. The value atan2(0, 1) = 0 multiplies a zero vector part,
    # so the returned log is the same origin either way.
    at_origin = (norm_v == 0) & (quaternion_scalar == 0)
    safe_scalar = torch.where(at_origin, torch.ones_like(quaternion_scalar), quaternion_scalar)
    theta = torch.atan2(norm_v, safe_scalar)

    quaternion_log: torch.Tensor = (quaternion_vector * theta / norm_q).to(orig_dtype)

    return quaternion_log


# based on:
# https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py#L138


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    r"""Convert axis-angle vectors ``(*, 3)`` in radians to ``(w, x, y, z)`` quaternions ``(*, 4)``.

    ``q = axis_angle_to_quaternion(axis_angle)`` takes rotation vectors ``(*, 3)``, each the unit axis scaled by the
    angle ``theta`` in radians, and returns the unit quaternions ``(cos(theta / 2), sin(theta / 2) * axis)`` of shape
    ``(*, 4)``, real part first. The sign is not canonicalised. For the reverse, use :func:`quaternion_to_axis_angle`;
    for the rotation matrix, :func:`axis_angle_to_rotation_matrix`; from the half-angle log vector,
    :func:`quaternion_log_to_exp`.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Convention:
        - the ``(w, x, y, z)`` layout is spelled out on
          :func:`~kornia.geometry.conversions.quaternion_to_rotation_matrix`
        - nothing is canonicalised, so ``pi < theta < 3 * pi`` returns the
          ``w < 0`` half of the double cover: ``[2 * pi, 0., 0.]`` gives
          ``[-1., 0., 0., 0.]``. The sign of ``w = cos(theta / 2)`` keeps
          alternating beyond that interval: ``[4 * pi, 0., 0.]`` gives
          ``[1., 0., 0., 0.]``
        - the round trip with
          :func:`~kornia.geometry.conversions.quaternion_to_axis_angle` holds
          to roundoff for ``0 <= theta <= pi``
        - ``angle_axis_to_quaternion`` is the deprecated alias of this function
          since 0.7.0
        - an integral or boolean input is **promoted**, not rejected: the
          ``sqrt`` on the way in already produces a floating tensor, and the
          output is allocated from that rather than from the input, so the
          result comes back at the ambient default floating dtype
        - differentiable at the zero rotation. The ``sqrt`` whose derivative is
          unbounded there never sees its zero radicand, so the gradient is the
          analytic limit of the surrounding map rather than ``nan``. The guard
          is elementwise, and away from zero it moves no forward bit

    Args:
        axis_angle: tensor with axis angle in radians.

    Return:
        tensor with quaternion.

    Shape:
        - Input: ``(*, 3)`` where `*` means, any number of dimensions
        - Output: ``(*, 4)``

    See Also:
        - :func:`quaternion_to_axis_angle`: the reverse conversion, quaternion to axis-angle.
        - :func:`axis_angle_to_rotation_matrix`: the rotation matrix of the same vector.
        - :func:`rotation_matrix_to_quaternion`: the quaternion of a rotation matrix instead, its sign not canonical.
        - :func:`quaternion_log_to_exp`: the same quaternion from half the vector, the exponential map.
        - :meth:`~kornia.geometry.quaternion.Quaternion.from_axis_angle`: the same quaternion as a
          :class:`~kornia.geometry.quaternion.Quaternion` object.

    Example:
        >>> axis_angle = torch.tensor((0., 1., 0.))
        >>> axis_angle_to_quaternion(axis_angle)
        tensor([0.8776, 0.0000, 0.4794, 0.0000])

    """
    if not isinstance(axis_angle, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(axis_angle)}")

    if not axis_angle.shape[-1] == 3:
        raise ValueError(f"Input must be a tensor of shape Nx3 or 3. Got {axis_angle.shape}")

    # unpack input and compute conversion
    a0: torch.Tensor = axis_angle[..., 0:1]
    a1: torch.Tensor = axis_angle[..., 1:2]
    a2: torch.Tensor = axis_angle[..., 2:3]
    theta_squared: torch.Tensor = a0 * a0 + a1 * a1 + a2 * a2

    mask: torch.Tensor = theta_squared > 0.0

    # the sqrt and the division below are only taken where the radicand is non zero. the inner
    # `where` keeps the zero rotation away from the sqrt so its backward pass stays finite.
    safe_theta_squared: torch.Tensor = torch.where(mask, theta_squared, torch.ones_like(theta_squared))
    theta: torch.Tensor = torch.where(mask, torch.sqrt(safe_theta_squared), torch.zeros_like(theta_squared))
    half_theta: torch.Tensor = theta * 0.5

    ones: torch.Tensor = torch.ones_like(half_theta)

    k_neg: torch.Tensor = 0.5 * ones
    k_pos: torch.Tensor = torch.sin(half_theta) / torch.where(mask, theta, ones)
    k: torch.Tensor = torch.where(mask, k_pos, k_neg)
    w: torch.Tensor = torch.where(mask, torch.cos(half_theta), ones)

    # the output dtype comes from the computed values, not from the input: an integer input is
    # promoted to floating point by the sqrt above, and writing that back into an integer buffer
    # would truncate every component to zero.
    quaternion: torch.Tensor = torch.zeros(size=(*axis_angle.shape[:-1], 4), dtype=w.dtype, device=axis_angle.device)
    quaternion[..., 1:2] = a0 * k
    quaternion[..., 2:3] = a1 * k
    quaternion[..., 3:4] = a2 * k
    quaternion[..., 0:1] = w
    return quaternion


@deprecated(replace_with="axis_angle_to_quaternion", version="0.7.0")
def angle_axis_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Deprecated: use :func:`~kornia.geometry.conversions.axis_angle_to_quaternion`.

    Each call emits a ``DeprecationWarning`` (deprecated since 0.7.0) and returns what the new name returns.
    """
    return axis_angle_to_quaternion(axis_angle)


# inspired by: https://stackoverflow.com/questions/56207448/efficient-quaternions-to-euler-transformation


def euler_from_quaternion(
    w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Convert a quaternion given as four tensors w, x, y, z to roll, pitch and yaw in radians.

    ``roll, pitch, yaw = euler_from_quaternion(w, x, y, z)`` takes the four quaternion components as **separate**
    tensors, not as one ``(*, 4)`` tensor, and returns three tensors of their shape: the angles in radians about ``x``,
    ``y`` and ``z`` of the rotation ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``, with ``roll`` and ``yaw`` in
    :math:`[-\pi, \pi]` and ``pitch`` in :math:`[-\pi/2, \pi/2]`. For a ``(*, 4)`` quaternion ``q``, pass
    ``*q.unbind(-1)``. For the reverse, use :func:`quaternion_from_euler`.

    Convention:
        - the four shapes must be exactly equal: broadcastable shapes such as
          ``(2,)`` and ``(1,)`` raise ``BaseError``
        - the composition the triple stands for is documented on
          :func:`~kornia.geometry.conversions.quaternion_from_euler`
        - away from ``|pitch| = pi/2`` it returns the triple of the input
          rotation folded into ``roll, yaw`` in ``[-pi, pi]`` — the interval is
          closed at both ends: they come from ``atan2``, which reaches ``±pi``
          with the endpoint's sign decided by signed zeros, so ``w = -0.0,
          x = 1.0, y = 0.0, z = -0.0`` returns roll exactly ``-pi`` while the
          ``+0.0`` twin returns ``+pi`` — and ``pitch`` in
          ``[-pi/2, pi/2]``, which inside those ranges is the input itself: the
          ``float64`` round trip of ``(0.3, 0.7, 1.1)`` through
          :func:`~kornia.geometry.conversions.quaternion_from_euler` returns to
          ``2.2e-16``, while ``(0.2, 2.5, 0.3)`` comes back as
          ``(-2.9416, 0.6416, -2.8416)`` — a different triple for the same
          rotation, to ``1.1e-16``
        - at ``|pitch| = pi/2`` (gimbal lock) ``roll`` and ``yaw`` are not individually
          determined, only ``yaw - roll`` at ``+pi/2`` and ``yaw + roll`` at ``-pi/2`` are, so the
          input triple cannot be recovered. It returns ``pitch`` exactly ``±pi/2``, ``roll`` exactly
          ``0`` and that one combination in ``yaw``, wrapped into ``[-pi, pi]``, which reproduces
          the rotation: the round trip through
          :func:`~kornia.geometry.conversions.quaternion_from_euler` is exact to rounding, and ``q``
          and ``-q`` return the same triple. Gimbal lock is detected as ``cos(pitch) < 2 *
          sqrt(eps)`` for the dtype's ``eps`` (``6.25e-2`` for ``float16``, ``1.77e-1`` for ``bfloat16``,
          ``6.9e-4`` for ``float32``, ``3.0e-8`` for ``float64``), so a pitch that close to ``±pi/2`` is
          also snapped to the pole and the returned rotation is off by at most its distance to it
        - the input is normalised with
          :func:`~kornia.geometry.conversions.normalize_quaternion` first, so a
          rescaled quaternion returns the same triple

    .. note::
        The gradient is finite everywhere, including exactly at gimbal lock. There ``roll`` and ``pitch``
        are constants, so their gradient is exactly ``0`` and all of it is in ``yaw``: for the quaternion of
        ``(roll, pitch, yaw) = (0.1, pi/2, 0.2)`` the gradient of ``roll + pitch + yaw`` with respect to
        ``(w, x, y, z)`` is ``(-0.141, -2.82, 0, 0)``. Within ``2 * sqrt(eps)`` of the pole it is the gradient
        of that constant-pitch triple, not of the plain formulas.

    Args:
        w: quaternion :math:`q_w` coefficient.
        x: quaternion :math:`q_x` coefficient.
        y: quaternion :math:`q_y` coefficient.
        z: quaternion :math:`q_z` coefficient.

    Return:
        A tuple with the euler angles ``roll``, ``pitch``, ``yaw`` in radians.

    See Also:
        - :func:`quaternion_from_euler`: the reverse conversion, Euler angles to quaternion components.
        - :meth:`~kornia.geometry.quaternion.Quaternion.to_euler`: the same angles of a
          :class:`~kornia.geometry.quaternion.Quaternion` object.

    """
    KORNIA_CHECK(w.shape == x.shape)
    KORNIA_CHECK(x.shape == y.shape)
    KORNIA_CHECK(y.shape == z.shape)

    quaternion = torch.stack((w, x, y, z), dim=-1)
    quaternion = normalize_quaternion(quaternion)
    w, x, y, z = quaternion.unbind(-1)

    yy = y * y

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + yy)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (yy + z * z)

    sinp = 2.0 * (w * y - z * x)
    sinp = sinp.clamp(min=-1.0, max=1.0)

    # Gimbal lock (kornia#3950). At |pitch| = pi/2 the factor cos(pitch) shared by the roll and yaw atan2
    # arguments vanishes, so both are atan2(0, 0) and say nothing about the rotation. Only one combination of
    # roll and yaw is determined there -- yaw - roll at +pi/2, yaw + roll at -pi/2 -- so the branch returns
    # roll = 0 and puts that combination in yaw, which reproduces the input rotation.
    #
    # cos(pitch)**2 is measured as sinr_cosp**2 + cosr_cosp**2, not as 1 - sinp**2: both arguments are
    # proportional to cos(pitch), so their sum of squares reads it directly. The sinp form cancels an
    # already-rounded sinp against itself, so at a true pole it reads k * eps for a k that depends on how the
    # inputs round and sits within a few ulps of any fixed threshold; it measurably misses ~0.3% of randomly
    # rescaled poles. The sum of squares reads ~eps**2 at a pole, 7 (float32) to 15 (float64) orders of
    # magnitude below the threshold. It is compared as a square, 4 * eps = (2 * sqrt(eps))**2, instead of through
    # torch.hypot, which the ONNX exporter cannot translate (prims.hypot, torch 2.14).
    #
    # The threshold is a compromise: a pitch inside it is snapped to the pole and so is off by its distance to
    # it, at most 2 * sqrt(eps) in cos(pitch) (6.9e-4 in float32, 3.0e-8 in float64). Replacing the 2 by
    # anything from about 1.0 to 2.8 passes the tests that pin both sides; larger factors snap pitches the plain
    # formulas still resolve better than the snap does (float32 pitch = pi/2 - 1e-3 is 3x worse at 4.0), and
    # smaller ones leave pitches near the pole unsnapped.
    cos_pitch_sq = sinr_cosp * sinr_cosp + cosr_cosp * cosr_cosp
    gimbal = cos_pitch_sq < 4.0 * torch.finfo(w.dtype).eps
    up = sinp > 0.0
    zero = torch.zeros_like(w)
    one = torch.ones_like(w)

    # Every atan2 below gets arguments that are never (0, 0): the pole arguments are replaced inside the branch,
    # and outside it cos(pitch) > 0. A torch.where around a plain atan2(0, 0) would still differentiate it, and
    # its derivative is nan on torch < 2.14 even under a zero cotangent (see AGENTS.md on safe arguments).
    # roll = atan2(0, 1) = 0 at the pole.
    roll = torch.where(gimbal, zero, sinr_cosp).atan2(torch.where(gimbal, one, cosr_cosp))

    # At the pole yaw = -+2 * atan2(x, w), written as atan2(2 * x * w, w * w - x * x) -- the same angle wrapped
    # into [-pi, pi], unchanged by q -> -q, and defined everywhere there because w * w + x * x = 1/2.
    double_sin = 2.0 * x * w
    yaw_num = torch.where(gimbal, torch.where(up, -double_sin, double_sin), siny_cosp)
    yaw_den = torch.where(gimbal, w * w - x * x, cosy_cosp)
    yaw = yaw_num.atan2(yaw_den)

    # d(asin)/dx = 1/sqrt(1-x^2) is unbounded at x = +-1 (gimbal lock), returning inf there on
    # every supported torch version; the clamp above bounds the value only, and passes the
    # gradient through on torch < 2.14 (2.14 zeroes it at the boundary, masking the defect).
    # Guard the gradient (kornia#4228) -- differentiate asin on a substituted safe argument, but take the
    # value from a detached copy at the real (possibly +-1) argument, so the returned pitch is
    # unchanged and only the gradient is finite.
    at_boundary = sinp.abs() >= 1.0
    safe_sinp = torch.where(at_boundary, torch.zeros_like(sinp), sinp)
    pitch = torch.where(at_boundary, sinp.detach().asin(), safe_sinp.asin())
    # Snap pitch to exactly +-pi/2 in the branch: asin of an argument that rounds just below 1 is off by
    # O(sqrt(eps)), which alone would leave the round trip that far from the input.
    half_pi = torch.full_like(pitch, math.pi / 2)
    pitch = torch.where(gimbal, torch.where(up, half_pi, -half_pi), pitch)

    return roll, pitch, yaw


def quaternion_from_euler(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert roll, pitch and yaw in radians to a quaternion returned as four tensors w, x, y, z.

    ``w, x, y, z = quaternion_from_euler(roll, pitch, yaw)`` takes the angles in radians about ``x``, ``y`` and ``z``
    as tensors of one shape and returns, as four tensors of that shape, the components of the unit quaternion of the
    rotation ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``, real part first. For one ``(*, 4)`` tensor, stack them with
    ``torch.stack((w, x, y, z), -1)``. For the reverse, use :func:`euler_from_quaternion`.

    Convention:
        - ``roll``, ``pitch`` and ``yaw`` are in **radians** and turn about
          ``x``, ``y`` and ``z`` respectively; the rotation they compose to is
          ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` — extrinsic X-Y-Z about the
          fixed axes, equivalently intrinsic Z-Y'-X''. The order matters: at
          ``(0.3, 0.7, 1.1)`` the returned quaternion's matrix differs from
          ``Rx @ Ry @ Rz`` by ``0.64``, from ``Ry @ Rz @ Rx`` by ``0.55`` and
          from ``Rx @ Rz @ Ry`` by ``0.25``
        - the three arguments must have exactly equal shapes; broadcastable
          shapes such as ``(2,)`` and ``(1,)`` raise ``BaseError``
        - the return is a tuple of **four separate tensors** ``(w, x, y, z)``,
          not a ``(*, 4)`` tensor: ``(0.3, 0.7, 1.1)`` gives
          ``[0.8186, -0.0575, 0.3624, 0.4418]``

    Args:
        roll: the roll euler angle.
        pitch: the pitch euler angle.
        yaw: the yaw euler angle.

    Return:
        A tuple with quaternion coefficients in order of ``wxyz``.

    See Also:
        - :func:`euler_from_quaternion`: the reverse conversion, quaternion components to Euler angles.
        - :meth:`~kornia.geometry.quaternion.Quaternion.from_euler`: the same quaternion as a
          :class:`~kornia.geometry.quaternion.Quaternion` object.

    """
    KORNIA_CHECK(roll.shape == pitch.shape)
    KORNIA_CHECK(pitch.shape == yaw.shape)

    roll_half = roll * 0.5
    pitch_half = pitch * 0.5
    yaw_half = yaw * 0.5

    cy = yaw_half.cos()
    sy = yaw_half.sin()
    cp = pitch_half.cos()
    sp = pitch_half.sin()
    cr = roll_half.cos()
    sr = roll_half.sin()

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    return qw, qx, qy, qz


# based on:
# https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L65-L71


def normalize_pixel_coordinates(
    pixel_coordinates: torch.Tensor, height: int, width: int, eps: float = 1e-8
) -> torch.Tensor:
    r"""Map 2D ``(x, y)`` pixel coordinates to ``[-1, 1]``, corner-aligned.

    ``coords_norm = normalize_pixel_coordinates(coords, height, width)`` takes pixel coordinates ``(*, 2)`` in
    ``(x, y)`` order and the image size in ``(height, width)`` order, and returns them as normalized
    :math:`[-1, 1]` coordinates, corner-aligned: the centres of the first and last pixel of each axis map to ``-1``
    and ``1``. This is the ``align_corners=True`` convention, and there is no other choice here: for the half-pixel
    convention of ``align_corners=False``, apply
    ``normal_transform_pixel(height, width, device=..., dtype=..., align_corners=False)``, on the device and in the
    floating dtype of the coordinates (cast integer ones first), with
    :func:`~kornia.geometry.linalg.transform_points`, which needs a point axis, ``(*, N, 2)`` (write a single ``(2,)``
    coordinate as ``coords[None]``). For the reverse, use
    :func:`denormalize_pixel_coordinates`; for ``(*, 3)`` voxel coordinates, :func:`normalize_pixel_coordinates3d`.

    Convention:
        See :doc:`Conventions & Pitfalls </get-started/conventions>` for the library-wide pixel-centre,
        normalized-coordinate and ``align_corners`` conventions.

        - ``pixel_coordinates`` is ``(*, 2)`` in ``(x, y)`` order: ``x``
          indexes columns and is scaled by ``width``, ``y`` indexes rows and is
          scaled by ``height``. The positional argument order is the other way
          round — ``(pixel_coordinates, height, width)``
        - the mapping is ``x_norm = 2 * x / (width - 1) - 1``: for
          ``width = 4`` the columns ``0``, ``1``, ``3`` map to ``-1``,
          ``-1/3``, ``+1``. :func:`torch.nn.functional.grid_sample` at its
          default ``align_corners=False`` would instead place the first and
          last of those values at pixels ``-0.5`` and ``3.5`` — half a pixel
          outside the image on each side — so ``align_corners=True`` must be
          passed explicitly when feeding this output to it. kornia's own
          :func:`~kornia.geometry.transform.remap` normalizes pixel maps with
          this function but resolves ``align_corners=None`` to ``False``
          (`#4504 <https://github.com/kornia/kornia/issues/4504>`_)
        - a singleton axis uses scale ``1`` and offset ``0``: its only valid
          pixel coordinate maps to the normalized centre, and the unit-scale
          extension keeps this function exactly invertible outside that lone
          valid coordinate. Zero and negative sizes raise ``ValueError``
        - the output is **not** clamped: coordinates outside the image
          extrapolate linearly past ``[-1, 1]``, as the example below shows

    .. note::
        Legacy ``torch.jit.trace`` graphs assume positive runtime sizes because
        tracing cannot retain the Python ``ValueError`` validation. Eager,
        TorchScript, and ``torch.compile`` calls validate sizes.

    .. deprecated:: 0.9.0
        ``eps`` no longer participates in the explicitly defined singleton-axis
        mapping. Passing a non-default value warns in eager mode; the parameter
        will be removed in a future breaking release.

    Args:
        pixel_coordinates: the grid with pixel coordinates. Shape can be ``(*, 2)``.
        height: the image height, the number of rows, which scales ``y``.
        width: the image width, the number of columns, which scales ``x``.
        eps: deprecated compatibility parameter. It is ignored.

    Return:
        the normalized pixel coordinates with shape ``(*, 2)``.

    See Also:
        - :func:`denormalize_pixel_coordinates`: the reverse map, :math:`[-1, 1]` back to pixels.
        - :func:`normalize_pixel_coordinates3d`: the 3D form, for ``(*, 3)`` coordinates in ``(d, x, y)`` order.
        - :func:`normal_transform_pixel`: the same map as a ``(1, 3, 3)`` matrix, with an ``align_corners`` choice.
        - :func:`~kornia.geometry.grid.create_meshgrid`: the whole ``(1, H, W, 2)`` pixel grid, normalized the same
          way by default.

    Examples:
        >>> coords = torch.tensor([[50., 100.]])
        >>> normalize_pixel_coordinates(coords, 100, 50)
        tensor([[1.0408, 1.0202]])

    """
    if pixel_coordinates.shape[-1] != 2:
        raise ValueError(f"Input pixel_coordinates must be of shape (*, 2). Got {pixel_coordinates.shape}")
    if not torch.jit.is_tracing() and (height <= 0 or width <= 0):
        raise ValueError(f"Input image size must be positive. Got height={height}, width={width}.")
    if not torch.jit.is_scripting() and not torch.jit.is_tracing() and not is_compiling() and eps != 1e-8:
        warnings.warn("`eps` is deprecated and ignored by `normalize_pixel_coordinates`.", FutureWarning, stacklevel=2)

    if torch.jit.is_scripting() or (not torch.jit.is_tracing() and not is_compiling()):
        sx = 1.0 if width == 1 else 2.0 / (width - 1.0)
        sy = 1.0 if height == 1 else 2.0 / (height - 1.0)
        tx = 0.0 if width == 1 else -1.0
        ty = 0.0 if height == 1 else -1.0
        if torch.jit.is_scripting():
            anchor = torch.ones((), device=pixel_coordinates.device, dtype=pixel_coordinates.dtype)
            factor = torch.stack([anchor * sx, anchor * sy])
            offset = torch.stack([anchor * tx, anchor * ty])
        else:
            work_dtype = pixel_coordinates.dtype if pixel_coordinates.is_floating_point() else torch.get_default_dtype()
            factor = torch.tensor([sx, sy], device=pixel_coordinates.device, dtype=work_dtype)
            offset = torch.tensor([tx, ty], device=pixel_coordinates.device, dtype=work_dtype)
    else:
        out_dtype = pixel_coordinates.dtype if pixel_coordinates.is_floating_point() else torch.get_default_dtype()
        # Low-precision floating types cannot represent every practical image size exactly
        # (e.g. bfloat16 rounds 257 to 256). Keep the symbolic size arithmetic in at least
        # float32, then cast the finished factors back to the coordinate dtype.
        work_dtype = torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
        width_t = torch.scalar_tensor(width, device=pixel_coordinates.device, dtype=work_dtype)
        height_t = torch.scalar_tensor(height, device=pixel_coordinates.device, dtype=work_dtype)
        one = torch.ones((), device=pixel_coordinates.device, dtype=work_dtype)
        zero = torch.zeros((), device=pixel_coordinates.device, dtype=work_dtype)
        factor = torch.stack(
            [
                torch.where(width_t == 1, one, 2.0 / (width_t - 1.0)),
                torch.where(height_t == 1, one, 2.0 / (height_t - 1.0)),
            ]
        )
        offset = torch.stack([torch.where(width_t == 1, zero, -one), torch.where(height_t == 1, zero, -one)])
        factor = factor.to(out_dtype)
        offset = offset.to(out_dtype)

    return factor * pixel_coordinates + offset


def denormalize_pixel_coordinates(
    pixel_coordinates: torch.Tensor, height: int, width: int, eps: float = 1e-8
) -> torch.Tensor:
    r"""Map 2D ``(x, y)`` coordinates from ``[-1, 1]`` back to pixels, corner-aligned.

    ``coords = denormalize_pixel_coordinates(coords_norm, height, width)`` is the inverse of
    :func:`normalize_pixel_coordinates`: it takes corner-aligned coordinates ``(*, 2)`` in ``(x, y)`` order, with
    ``-1`` and ``1`` at the centres of the first and last pixel of each axis, and the image size in
    ``(height, width)`` order, and returns the pixel coordinates ``(*, 2)``, ``x = (width - 1) * (x_norm + 1) / 2``,
    so that ``1`` maps to ``x = width - 1``. For ``(*, 3)`` voxel coordinates, use
    :func:`denormalize_pixel_coordinates3d`.

    Convention:
        See :doc:`Conventions & Pitfalls </get-started/conventions>` for the library-wide pixel-centre,
        normalized-coordinate and ``align_corners`` conventions.

        - singleton axes, non-positive-size validation, tracing assumptions, and
          the deprecated ``eps`` parameter follow the normalization function

    Args:
        pixel_coordinates: the normalized grid coordinates. Shape can be ``(*, 2)``.
        height: the image height, the number of rows, which scales ``y``.
        width: the image width, the number of columns, which scales ``x``.
        eps: deprecated compatibility parameter. It is ignored.

    Return:
        the denormalized pixel coordinates with shape ``(*, 2)``.

    See Also:
        - :func:`normalize_pixel_coordinates`: the forward map, pixels to :math:`[-1, 1]`.
        - :func:`denormalize_pixel_coordinates3d`: the 3D form, for ``(*, 3)`` coordinates in ``(d, x, y)`` order.

    Examples:
        >>> coords = torch.tensor([[-1., -1.]])
        >>> denormalize_pixel_coordinates(coords, 100, 50)
        tensor([[0., 0.]])

    """
    if pixel_coordinates.shape[-1] != 2:
        raise ValueError(f"Input pixel_coordinates must be of shape (*, 2). Got {pixel_coordinates.shape}")
    if not torch.jit.is_tracing() and (height <= 0 or width <= 0):
        raise ValueError(f"Input image size must be positive. Got height={height}, width={width}.")
    if not torch.jit.is_scripting() and not torch.jit.is_tracing() and not is_compiling() and eps != 1e-8:
        warnings.warn(
            "`eps` is deprecated and ignored by `denormalize_pixel_coordinates`.",
            FutureWarning,
            stacklevel=2,
        )

    if torch.jit.is_scripting() or (not torch.jit.is_tracing() and not is_compiling()):
        sx = 1.0 if width == 1 else (width - 1.0) / 2.0
        sy = 1.0 if height == 1 else (height - 1.0) / 2.0
        tx = 0.0 if width == 1 else sx
        ty = 0.0 if height == 1 else sy
        if torch.jit.is_scripting():
            anchor = torch.ones((), device=pixel_coordinates.device, dtype=pixel_coordinates.dtype)
            factor = torch.stack([anchor * sx, anchor * sy])
            offset = torch.stack([anchor * tx, anchor * ty])
        else:
            work_dtype = pixel_coordinates.dtype if pixel_coordinates.is_floating_point() else torch.get_default_dtype()
            factor = torch.tensor([sx, sy], device=pixel_coordinates.device, dtype=work_dtype)
            offset = torch.tensor([tx, ty], device=pixel_coordinates.device, dtype=work_dtype)
    else:
        out_dtype = pixel_coordinates.dtype if pixel_coordinates.is_floating_point() else torch.get_default_dtype()
        # Low-precision floating types cannot represent every practical image size exactly
        # (e.g. bfloat16 rounds 257 to 256). Keep the symbolic size arithmetic in at least
        # float32, then cast the finished factors back to the coordinate dtype.
        work_dtype = torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
        width_t = torch.scalar_tensor(width, device=pixel_coordinates.device, dtype=work_dtype)
        height_t = torch.scalar_tensor(height, device=pixel_coordinates.device, dtype=work_dtype)
        one = torch.ones((), device=pixel_coordinates.device, dtype=work_dtype)
        zero = torch.zeros((), device=pixel_coordinates.device, dtype=work_dtype)
        sx_t = torch.where(width_t == 1, one, (width_t - 1.0) / 2.0)
        sy_t = torch.where(height_t == 1, one, (height_t - 1.0) / 2.0)
        factor = torch.stack([sx_t, sy_t])
        offset = torch.stack([torch.where(width_t == 1, zero, sx_t), torch.where(height_t == 1, zero, sy_t)])
        factor = factor.to(out_dtype)
        offset = offset.to(out_dtype)

    return factor * pixel_coordinates + offset


def normalize_pixel_coordinates3d(
    pixel_coordinates: torch.Tensor, depth: int, height: int, width: int, eps: float = 1e-8
) -> torch.Tensor:
    r"""Map 3D ``(d, x, y)`` voxel coordinates to ``[-1, 1]``, corner-aligned.

    ``coords_norm = normalize_pixel_coordinates3d(coords, depth, height, width)`` takes voxel coordinates ``(*, 3)`` in
    ``(d, x, y)`` order, the order of :func:`~kornia.geometry.grid.create_meshgrid3d`, and returns them as
    normalized :math:`[-1, 1]` coordinates, corner-aligned and unclamped in each axis as
    :func:`normalize_pixel_coordinates` is.
    :func:`normal_transform_pixel3d` reads ``(x, y, z)`` instead, so a grid built for one permutes the axes when fed
    to the other. For the reverse, use :func:`denormalize_pixel_coordinates3d`.

    Convention:
        - ``pixel_coordinates`` is ``(*, 3)`` in ``(d, x, y)`` order —
          **depth first, then x, then y**, *not* ``(x, y, z)``. The
          three components are scaled by ``depth - 1``, ``width - 1`` and
          ``height - 1`` respectively, so with ``depth=3, height=5, width=9``
          the coordinate ``[2., 8., 4.]`` maps to ``[1., 1., 1.]``
        - the positional argument order is
          ``(pixel_coordinates, depth, height, width)``
        - singleton axes, non-positive-size validation, tracing assumptions, and
          the deprecated ``eps`` parameter follow
          :func:`~kornia.geometry.conversions.normalize_pixel_coordinates`

    Args:
        pixel_coordinates: the grid with pixel coordinates. Shape can be ``(*, 3)``.
        depth: the volume depth, which scales ``d``.
        height: the volume height, which scales ``y``.
        width: the volume width, which scales ``x``.
        eps: deprecated compatibility parameter. It is ignored.

    Return:
        the normalized pixel coordinates, ``(*, 3)`` in ``(d, x, y)`` order.

    See Also:
        - :func:`denormalize_pixel_coordinates3d`: the reverse map, :math:`[-1, 1]` back to voxels.
        - :func:`normalize_pixel_coordinates`: the 2D form, for ``(*, 2)`` coordinates in ``(x, y)`` order.
        - :func:`normal_transform_pixel3d`: the same scaling as a ``(1, 4, 4)`` matrix on ``(x, y, z, 1)``.

    """
    if pixel_coordinates.shape[-1] != 3:
        raise ValueError(f"Input pixel_coordinates must be of shape (*, 3). Got {pixel_coordinates.shape}")
    if not torch.jit.is_tracing() and (depth <= 0 or height <= 0 or width <= 0):
        raise ValueError(f"Input image size must be positive. Got depth={depth}, height={height}, width={width}.")
    if not torch.jit.is_scripting() and not torch.jit.is_tracing() and not is_compiling() and eps != 1e-8:
        warnings.warn(
            "`eps` is deprecated and ignored by `normalize_pixel_coordinates3d`.",
            FutureWarning,
            stacklevel=2,
        )

    if torch.jit.is_scripting() or (not torch.jit.is_tracing() and not is_compiling()):
        sd = 1.0 if depth == 1 else 2.0 / (depth - 1.0)
        sx = 1.0 if width == 1 else 2.0 / (width - 1.0)
        sy = 1.0 if height == 1 else 2.0 / (height - 1.0)
        td = 0.0 if depth == 1 else -1.0
        tx = 0.0 if width == 1 else -1.0
        ty = 0.0 if height == 1 else -1.0
        if torch.jit.is_scripting():
            anchor = torch.ones((), device=pixel_coordinates.device, dtype=pixel_coordinates.dtype)
            factor = torch.stack([anchor * sd, anchor * sx, anchor * sy])
            offset = torch.stack([anchor * td, anchor * tx, anchor * ty])
        else:
            work_dtype = pixel_coordinates.dtype if pixel_coordinates.is_floating_point() else torch.get_default_dtype()
            factor = torch.tensor([sd, sx, sy], device=pixel_coordinates.device, dtype=work_dtype)
            offset = torch.tensor([td, tx, ty], device=pixel_coordinates.device, dtype=work_dtype)
    else:
        out_dtype = pixel_coordinates.dtype if pixel_coordinates.is_floating_point() else torch.get_default_dtype()
        # Low-precision floating types cannot represent every practical image size exactly
        # (e.g. bfloat16 rounds 257 to 256). Keep the symbolic size arithmetic in at least
        # float32, then cast the finished factors back to the coordinate dtype.
        work_dtype = torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
        depth_t = torch.scalar_tensor(depth, device=pixel_coordinates.device, dtype=work_dtype)
        width_t = torch.scalar_tensor(width, device=pixel_coordinates.device, dtype=work_dtype)
        height_t = torch.scalar_tensor(height, device=pixel_coordinates.device, dtype=work_dtype)
        one = torch.ones((), device=pixel_coordinates.device, dtype=work_dtype)
        zero = torch.zeros((), device=pixel_coordinates.device, dtype=work_dtype)
        factor = torch.stack(
            [
                torch.where(depth_t == 1, one, 2.0 / (depth_t - 1.0)),
                torch.where(width_t == 1, one, 2.0 / (width_t - 1.0)),
                torch.where(height_t == 1, one, 2.0 / (height_t - 1.0)),
            ]
        )
        offset = torch.stack(
            [
                torch.where(depth_t == 1, zero, -one),
                torch.where(width_t == 1, zero, -one),
                torch.where(height_t == 1, zero, -one),
            ]
        )
        factor = factor.to(out_dtype)
        offset = offset.to(out_dtype)

    return factor * pixel_coordinates + offset


def denormalize_pixel_coordinates3d(
    pixel_coordinates: torch.Tensor, depth: int, height: int, width: int, eps: float = 1e-8
) -> torch.Tensor:
    r"""Map 3D ``(d, x, y)`` coordinates from ``[-1, 1]`` back to voxels, corner-aligned.

    ``coords = denormalize_pixel_coordinates3d(coords_norm, depth, height, width)`` is the inverse of
    :func:`normalize_pixel_coordinates3d`: it takes corner-aligned coordinates ``(*, 3)`` in ``(d, x, y)`` order and
    returns the voxel coordinates ``(*, 3)`` in the same order. For ``(*, 2)`` pixel coordinates, use
    :func:`denormalize_pixel_coordinates`.

    Convention:
        - the ``(d, x, y)`` component order and the
          ``(pixel_coordinates, depth, height, width)`` positional order are
          those of the normalization: with ``depth=3, height=5, width=9`` the
          input ``[0., 0., 0.]`` maps back to ``[1., 4., 2.]``
        - singleton axes, non-positive-size validation, tracing assumptions, and
          the deprecated ``eps`` parameter follow the 2-D counterpart

    Args:
        pixel_coordinates: the normalized grid coordinates. Shape can be ``(*, 3)``.
        depth: the volume depth, which scales ``d``.
        height: the volume height, which scales ``y``.
        width: the volume width, which scales ``x``.
        eps: deprecated compatibility parameter. It is ignored.

    Return:
        the denormalized pixel coordinates, ``(*, 3)`` in ``(d, x, y)`` order.

    See Also:
        - :func:`normalize_pixel_coordinates3d`: the forward map, voxels to :math:`[-1, 1]`.
        - :func:`denormalize_pixel_coordinates`: the 2D form, for ``(*, 2)`` coordinates in ``(x, y)`` order.

    """
    if pixel_coordinates.shape[-1] != 3:
        raise ValueError(f"Input pixel_coordinates must be of shape (*, 3). Got {pixel_coordinates.shape}")
    if not torch.jit.is_tracing() and (depth <= 0 or height <= 0 or width <= 0):
        raise ValueError(f"Input image size must be positive. Got depth={depth}, height={height}, width={width}.")
    if not torch.jit.is_scripting() and not torch.jit.is_tracing() and not is_compiling() and eps != 1e-8:
        warnings.warn(
            "`eps` is deprecated and ignored by `denormalize_pixel_coordinates3d`.",
            FutureWarning,
            stacklevel=2,
        )

    if torch.jit.is_scripting() or (not torch.jit.is_tracing() and not is_compiling()):
        sd = 1.0 if depth == 1 else (depth - 1.0) / 2.0
        sx = 1.0 if width == 1 else (width - 1.0) / 2.0
        sy = 1.0 if height == 1 else (height - 1.0) / 2.0
        td = 0.0 if depth == 1 else sd
        tx = 0.0 if width == 1 else sx
        ty = 0.0 if height == 1 else sy
        if torch.jit.is_scripting():
            anchor = torch.ones((), device=pixel_coordinates.device, dtype=pixel_coordinates.dtype)
            factor = torch.stack([anchor * sd, anchor * sx, anchor * sy])
            offset = torch.stack([anchor * td, anchor * tx, anchor * ty])
        else:
            work_dtype = pixel_coordinates.dtype if pixel_coordinates.is_floating_point() else torch.get_default_dtype()
            factor = torch.tensor([sd, sx, sy], device=pixel_coordinates.device, dtype=work_dtype)
            offset = torch.tensor([td, tx, ty], device=pixel_coordinates.device, dtype=work_dtype)
    else:
        out_dtype = pixel_coordinates.dtype if pixel_coordinates.is_floating_point() else torch.get_default_dtype()
        # Low-precision floating types cannot represent every practical image size exactly
        # (e.g. bfloat16 rounds 257 to 256). Keep the symbolic size arithmetic in at least
        # float32, then cast the finished factors back to the coordinate dtype.
        work_dtype = torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
        depth_t = torch.scalar_tensor(depth, device=pixel_coordinates.device, dtype=work_dtype)
        width_t = torch.scalar_tensor(width, device=pixel_coordinates.device, dtype=work_dtype)
        height_t = torch.scalar_tensor(height, device=pixel_coordinates.device, dtype=work_dtype)
        one = torch.ones((), device=pixel_coordinates.device, dtype=work_dtype)
        zero = torch.zeros((), device=pixel_coordinates.device, dtype=work_dtype)
        sd_t = torch.where(depth_t == 1, one, (depth_t - 1.0) / 2.0)
        sx_t = torch.where(width_t == 1, one, (width_t - 1.0) / 2.0)
        sy_t = torch.where(height_t == 1, one, (height_t - 1.0) / 2.0)
        factor = torch.stack([sd_t, sx_t, sy_t])
        offset = torch.stack(
            [
                torch.where(depth_t == 1, zero, sd_t),
                torch.where(width_t == 1, zero, sx_t),
                torch.where(height_t == 1, zero, sy_t),
            ]
        )
        factor = factor.to(out_dtype)
        offset = offset.to(out_dtype)

    return factor * pixel_coordinates + offset


def angle_to_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    r"""Build 2D rotation matrices ``(*, 2, 2)`` from angles in degrees, counter-clockwise as displayed.

    ``R = angle_to_rotation_matrix(angle)`` takes angles of any shape ``(*)`` in **degrees** and returns the matrices
    ``[[cos, sin], [-sin, cos]]`` of shape ``(*, 2, 2)``. Applied to ``(x, y)`` column vectors in image coordinates,
    with ``y`` down, a positive angle turns counter-clockwise as displayed, which is clockwise in a ``y``-up plot. For
    a ``(B, 2, 3)`` matrix that also scales and turns about a centre, use
    :func:`~kornia.geometry.transform.get_rotation_matrix2d`; for an angle in radians, pass ``rad2deg(angle)``.

    Convention:
        - the matrix has ``det = +1``; for
          ``angle = 30`` it is ``[[0.8660, 0.5000], [-0.5000, 0.8660]]``
        - applied on the left to a column vector ``(x, y)``, ``angle = 30`` maps
          ``(1., 0.)`` to ``(0.8660, -0.5000)`` — counter-clockwise **as
          displayed** under kornia's y-down image axes. In the raw coordinate
          plane this is the opposite sense to
          :func:`~kornia.geometry.conversions.cart2pol`, whose Convention block
          spells out the modulo-:math:`2\pi` relation between the two ops

    Args:
        angle: tensor of angles in degrees, any shape ``(*)``.

    Returns:
        tensor of rotation matrices with shape ``(*, 2, 2)``.

    See Also:
        - :func:`~kornia.geometry.transform.get_rotation_matrix2d`: the ``(B, 2, 3)`` matrix that also scales and
          turns about a centre, in the same sense.
        - :func:`axis_angle_to_rotation_matrix`: a 3D ``(*, 3, 3)`` rotation from an axis-angle vector in radians.
        - :func:`cart2pol`: the polar angle of a point, which this rotation decreases.

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = angle_to_rotation_matrix(input)  # Nx3x2x2

    """
    ang_rad = deg2rad(angle)
    cos_a: torch.Tensor = torch.cos(ang_rad)
    sin_a: torch.Tensor = torch.sin(ang_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)


def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: tuple[int, int],
    dsize_dst: tuple[int, int],
    align_corners: bool = True,
) -> torch.Tensor:
    r"""Express a source→destination pixel homography ``(B, 3, 3)`` in ``[-1, 1]`` coordinates.

    ``H_norm = normalize_homography(H, dsize_src, dsize_dst)`` takes homographies ``(B, 3, 3)``, or one ``(3, 3)``,
    that map **source pixels to destination pixels**, and the ``(height, width)`` of both images, and returns
    ``N_dst @ H @ inv(N_src)`` ``(B, 3, 3)``: the same maps between normalized :math:`[-1, 1]` coordinates, with ``N``
    from :func:`normal_transform_pixel`. Pass the ``align_corners`` of the sampling that will use the result: for a
    ``(B, 3, 3)`` ``H`` and ``H_norm = normalize_homography(H, img.shape[-2:], dsize, align_corners=a)``,
    ``homography_warp(img, torch.linalg.inv(H_norm), dsize, align_corners=a)`` samples what
    ``warp_perspective(img, H, dsize, align_corners=a)`` samples, since
    :func:`~kornia.geometry.transform.warp_perspective` normalizes a pixel homography itself. For the reverse, use
    :func:`denormalize_homography`; for ``(B, 4, 4)`` voxel homographies, :func:`normalize_homography3d`.

    Convention:
        - ``dsize_src`` drives the right factor and ``dsize_dst`` the left one:
          a ``+2`` pixel shift in ``x`` on a 5-wide image comes back as ``+1.0``
          (``+0.8`` with ``align_corners=False``).
          :func:`~kornia.geometry.conversions.denormalize_homography` computes
          the reverse ``inv(N_dst) @ H @ N_src``
        - both ``dsize`` arguments are ``(height, width)``, while the matrix acts
          on ``(x, y, 1)`` column vectors
        - a ``(3, 3)`` or ``(B, 3, 3)`` matrix is accepted and batching is
          per sample; an unbatched input is returned as ``(1, 3, 3)``, and any
          other shape raises ``ValueError``
        - in eager mode ``inv(N_src)`` is a closed-form adjugate built from
          ``torch.linalg.cross``, so no ``cusolver`` is needed. A ``float16`` or
          ``bfloat16`` matrix is inverted in ``float32`` and the result cast back:
          the ``float16`` determinant of a large image's normalization matrix is
          subnormal or zero, and some backends lack a ``bfloat16`` ``cross`` kernel
          (MPS on torch 2.5.1). For any other dtype, a backend without a ``cross``
          kernel for it raises here

    .. warning::
        An integer matrix is outside the contract and is not rejected: the
        normalization matrices are cast to it and truncated, and the call then
        fails inside torch or returns ``nan``, depending on the backend.

    .. warning::
        The :math:`[-1, 1]` frames follow this function's own ``align_corners``,
        which defaults to ``True`` (corner-aligned) — **not** the ``False`` that
        :py:func:`torch.nn.functional.grid_sample` and
        :func:`~kornia.geometry.transform.homography_warp` default to. The
        value passed here must match the ``grid_sample`` call that consumes the
        result; a mismatch applies a spurious sub-pixel scale and shift.
        :func:`~kornia.geometry.transform.warp_perspective`,
        :func:`~kornia.geometry.transform.warp_affine` and
        :func:`~kornia.geometry.transform.homography_warp` forward their own
        flag, so they are consistent under both settings; see
        `#3904 <https://github.com/kornia/kornia/issues/3904>`_.
        :func:`~kornia.geometry.conversions.normalize_homography3d` still has
        no ``align_corners`` parameter and is corner-aligned unconditionally, so
        :func:`~kornia.geometry.transform.warp_affine3d` keeps this mismatch at
        ``align_corners=False``. Tracked in
        `#4503 <https://github.com/kornia/kornia/issues/4503>`_.

    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          normalized. ``(B, 3, 3)``, or an unbatched ``(3, 3)``, which
          is promoted to ``(1, 3, 3)``
        dsize_src: size of the source image (height, width).
        dsize_dst: size of the destination image (height, width).
        align_corners: which :py:func:`torch.nn.functional.grid_sample` convention the
          normalized :math:`[-1, 1]` coordinates follow, forwarded to
          :func:`normal_transform_pixel`. Must match the ``align_corners`` of the
          ``grid_sample`` call that ultimately consumes the result, otherwise the warp
          picks up a spurious sub-pixel scale and shift.

    Returns:
        the normalized homography of shape ``(B, 3, 3)``.

    See Also:
        - :func:`denormalize_homography`: the reverse, a normalized homography back to pixels.
        - :func:`normalize_homography3d`: the 3D form for ``(B, 4, 4)`` voxel homographies, always corner-aligned.
        - :func:`normal_transform_pixel`: the ``(1, 3, 3)`` normalization matrix ``N`` composed here.
        - :func:`~kornia.geometry.transform.warp_perspective`: warps by the pixel homography, normalizing it with
          this function.
        - :func:`~kornia.geometry.transform.homography_warp`: warps by the inverse of the result, a
          destination→source normalized homography.

    """
    if not isinstance(dst_pix_trans_src_pix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(dst_pix_trans_src_pix)}")

    if not (dst_pix_trans_src_pix.ndim in (2, 3) and dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {dst_pix_trans_src_pix.shape}")

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    normalization_dtype = (
        dst_pix_trans_src_pix.dtype
        if dst_pix_trans_src_pix.is_floating_point() or dst_pix_trans_src_pix.is_complex()
        else None
    )
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(
        src_h,
        src_w,
        device=dst_pix_trans_src_pix.device,
        dtype=normalization_dtype,
        align_corners=align_corners,
    ).to(dst_pix_trans_src_pix)

    # Closed-form 3x3 inverse of the (well-conditioned) pixel-normalization matrix: cusolver-free,
    # so homography normalization runs on the Jetson wheel where ``torch.linalg.inv`` dlopen-fails.
    src_pix_trans_src_norm = _inverse_3x3_closed_form(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(
        dst_h,
        dst_w,
        device=dst_pix_trans_src_pix.device,
        dtype=normalization_dtype,
        align_corners=align_corners,
    ).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    align_corners: bool = True,
) -> torch.Tensor:
    r"""Compute the ``(1, 3, 3)`` matrix that maps 2D pixel coordinates ``(x, y)`` to ``[-1, 1]``.

    ``N = normal_transform_pixel(height, width)`` returns a **single** matrix ``(1, 3, 3)``, never batched, that maps
    the homogeneous pixel coordinates ``(x, y, 1)`` of a ``height`` by ``width`` image to :math:`[-1, 1]`: the
    corner-aligned map of :func:`normalize_pixel_coordinates` by default, the half-pixel map with
    ``align_corners=False``. It is the ``N`` that :func:`normalize_homography` and :func:`denormalize_homography`
    compose with a homography. To normalize points directly, use :func:`normalize_pixel_coordinates`; for
    ``(1, 4, 4)`` volume matrices, :func:`normal_transform_pixel3d`.

    Convention:
        - ``x`` indexes columns and is scaled by ``width``, ``y`` indexes rows
          and is scaled by ``height``. The positional argument order is the
          other way round, ``(height, width)``
        - the mapping is **corner-aligned by default**: scale ``2 / (size - 1)``,
          offset ``-1``, so the pixel *centres* ``0`` and ``size - 1`` map to
          exactly ``-1`` and ``+1``. ``align_corners=False`` selects the
          half-pixel mapping instead; see the ``align_corners`` bullet below.
          ``normal_transform_pixel(4, 5)`` has rows
          ``[0.5, 0.0, -1.0]`` and ``[0.0, 0.6667, -1.0]``, and sends the pixels
          ``(0, 0)``, ``(4, 3)`` and ``(2, 1.5)`` to ``(-1, -1)``, ``(1, 1)``
          and ``(0, 0)``
        - for sizes ``>= 2`` this is the same map as
          :func:`~kornia.geometry.conversions.normalize_pixel_coordinates`. In
          ``float32``/``float64`` the two routes agree to the working dtype's
          rounding; in ``float16``/``bfloat16`` they can differ by about one
          ``eps``, because the matrix is applied through a matmul while the
          helper rounds elementwise
        - the convention is selected by ``align_corners`` (``True`` by
          default). ``False`` is the half-pixel mapping with scale ``2 / size``
          and offset ``1 / size - 1``, so that :math:`\pm 1` fall on the outer
          pixel *edges*. :func:`~kornia.geometry.conversions.normalize_homography`
          and :func:`~kornia.geometry.conversions.denormalize_homography` forward
          their own ``align_corners`` here, while
          :func:`~kornia.geometry.conversions.normal_transform_pixel3d` is still
          corner-aligned unconditionally; see the convention warning there
        - a singleton axis maps its only pixel to the centre of the normalized
          range: under ``align_corners=True`` with scale ``1`` and offset ``0``,
          which keeps the matrix invertible for homography composition. Zero
          and negative sizes raise ``ValueError``
        - with ``dtype=None`` the matrix takes ``torch.get_default_dtype()``
          (``float32`` unless the default was changed); an integer or bool
          ``dtype`` raises ``ValueError``, since the fractional scales would
          truncate to ``0``, for every size and even under ``disable_checks()``.
          Every floating and complex dtype is accepted

    .. note::
        Legacy ``torch.jit.trace`` graphs assume positive runtime sizes because
        tracing cannot retain the Python ``ValueError`` validation. Eager,
        TorchScript, and ``torch.compile`` calls validate sizes.

    .. deprecated:: 0.9.0
        ``eps`` no longer participates in the explicitly defined singleton-axis
        mapping. Passing a non-default value warns in eager mode; the parameter
        will be removed in a future breaking release.

    Args:
        height: image height.
        width: image width.
        eps: deprecated compatibility parameter. It is ignored.
        device: device to place the result on.
        dtype: dtype of the result. ``None`` means ``torch.get_default_dtype()``.
        align_corners: which :py:func:`torch.nn.functional.grid_sample` convention to
          normalize to. ``True`` maps pixel centers :math:`[0, size-1]` to
          :math:`[-1, 1]`; ``False`` uses the half-pixel mapping
          :math:`x_{norm} = (2x + 1) / W - 1`, where :math:`\pm 1` are the outer pixel
          *edges*.

    Returns:
        normalized transform with shape ``(1, 3, 3)``.

    See Also:
        - :func:`normal_transform_pixel3d`: the 3D ``(1, 4, 4)`` matrix, always corner-aligned.
        - :func:`normalize_pixel_coordinates`: the same corner-aligned map applied to ``(*, 2)`` points.
        - :func:`normalize_homography`: conjugates a pixel homography with this matrix.

    Example:
        >>> normal_transform_pixel(4, 5)
        tensor([[[ 0.5000,  0.0000, -1.0000],
                 [ 0.0000,  0.6667, -1.0000],
                 [ 0.0000,  0.0000,  1.0000]]])

    """
    if not torch.jit.is_tracing() and (height <= 0 or width <= 0):
        raise ValueError(f"Input image size must be positive. Got height={height}, width={width}.")
    # A normalization matrix scales by 2/(size - 1), which is fractional for every
    # dimension larger than 3 pixels. An integer dtype truncates those scales to 0 and
    # yields a rank-deficient matrix that maps the whole image to a single point. The
    # accepted dtypes are a property of the function rather than of height and width, so
    # the rejection is uniform: a dimension of 3 pixels or fewer makes the scale integral
    # and would survive, and accepting integers only there would make the domain depend
    # on the image.
    # Raised directly rather than through KORNIA_CHECK, which disable_checks(),
    # python -O and KORNIA_CHECKS=0 all switch off: a domain restriction whose
    # alternative is silently wrong output cannot be optional.
    # A zero-element probe rather than dtype.is_floating_point, because TorchScript cannot
    # access dtype attributes and this runs inside the scripted warp_affine path. The probe
    # answers for any dtype, including the float8 formats and whatever torch adds next. The
    # membership test in front of it is a fast path, not the domain, and it carries the
    # complex dtypes so that a `complex32` call does not emit torch's experimental-ComplexHalf
    # warning from an allocation the caller never asked for.
    if dtype is not None and dtype not in (
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.complex32,
        torch.complex64,
        torch.complex128,
    ):
        probe = torch.empty(0, dtype=dtype)
        if not (probe.is_floating_point() or probe.is_complex()):
            raise ValueError(f"dtype must be a floating point or complex type. Got {dtype}.")
    if not torch.jit.is_scripting() and not torch.jit.is_tracing() and not is_compiling() and eps != 1e-14:
        warnings.warn("`eps` is deprecated and ignored by `normal_transform_pixel`.", FutureWarning, stacklevel=2)

    if torch.jit.is_scripting() or (not torch.jit.is_tracing() and not is_compiling()):
        # Eager and TorchScript take the scalar branch, which is an order of magnitude
        # cheaper on this hot path. Graph capture takes the tensor form below so symbolic
        # sizes retain the singleton decision.
        if align_corners:
            sx = 1.0 if width == 1 else 2.0 / (width - 1.0)
            sy = 1.0 if height == 1 else 2.0 / (height - 1.0)
            tx = 0.0 if width == 1 else -1.0
            ty = 0.0 if height == 1 else -1.0
        else:
            # Half-pixel mapping. It is finite for a size of 1 and lands that pixel on the
            # centre (scale 2, offset 0), so it needs no singleton special case. The offset is
            # written as a single division rather than ``1 / size - 1`` so that the graph-capture
            # branch below, which evaluates it in float32 instead of Python doubles, rounds once
            # and stays bit-identical to this one; the two-step form differs by one float32 step
            # at 5 of the first 4999 sizes.
            sx = 2.0 / width
            sy = 2.0 / height
            tx = (1.0 - width) / width
            ty = (1.0 - height) / height
        if torch.jit.is_scripting():
            # TorchScript builds ``torch.tensor`` from a float list in float32 even when ``dtype`` is
            # float64, so scale a typed anchor instead to keep the constants in the output dtype.
            anchor = torch.ones((), device=device, dtype=dtype)
            zero = anchor * 0.0
            tr_mat = torch.stack(
                [
                    torch.stack([anchor * sx, zero, anchor * tx]),
                    torch.stack([zero, anchor * sy, anchor * ty]),
                    torch.stack([zero, zero, anchor]),
                ]
            )
        elif dtype is None or dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            # Filled on the device: ``torch.tensor(..., device=cuda)`` copies from the host and waits for every
            # queued kernel before it returns. The values round from the same doubles to ``dtype`` either way.
            tr_mat = _constant_tensor([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        else:
            # Complex and float8 dtypes keep the host literal: ``torch.full`` has no float8 fill on torch 2.5.1 (CPU)
            # or on MPS.
            tr_mat = torch.tensor([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
    else:
        # Low-precision floating types cannot represent every practical image size exactly
        # (e.g. bfloat16 rounds 257 to 256). Keep symbolic size arithmetic in at least
        # float32, then cast the finished matrix to the requested output dtype. Resolve that
        # output dtype FIRST: a ``dtype=None`` call inherits the default floating dtype, which
        # may itself be a half type, so promoting only an *explicit* half ``dtype`` would leave
        # ``torch.set_default_dtype(torch.float16)`` rounding the size below.
        out_dtype = dtype if dtype is not None else torch.get_default_dtype()
        work_dtype = out_dtype if out_dtype in (torch.float32, torch.float64) else torch.float32
        width_t = torch.scalar_tensor(width, device=device, dtype=work_dtype)
        height_t = torch.scalar_tensor(height, device=device, dtype=work_dtype)
        one = torch.ones((), device=device, dtype=work_dtype)
        zero = torch.zeros((), device=device, dtype=work_dtype)

        if align_corners:
            # A singleton axis has no extent. Map its only pixel to the normalized centre
            # while keeping the homogeneous transform invertible for homography composition.
            sx_t = torch.where(width_t == 1, one, 2.0 / (width_t - 1.0))
            sy_t = torch.where(height_t == 1, one, 2.0 / (height_t - 1.0))
            tx_t = torch.where(width_t == 1, zero, -one)
            ty_t = torch.where(height_t == 1, zero, -one)
        else:
            # Single division, matching the scalar branch above bit for bit.
            sx_t = 2.0 / width_t
            sy_t = 2.0 / height_t
            tx_t = (one - width_t) / width_t
            ty_t = (one - height_t) / height_t

        # Construct the matrix in one shot (no in-place mutation).
        tr_mat = torch.stack(
            [torch.stack([sx_t, zero, tx_t]), torch.stack([zero, sy_t, ty_t]), torch.stack([zero, zero, one])]
        ).to(dtype=out_dtype)  # 3x3

    return tr_mat.unsqueeze(0)  # 1x3x3


def normal_transform_pixel3d(
    depth: int,
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Compute the ``(1, 4, 4)`` matrix that maps 3D voxel coordinates ``(x, y, z)`` to ``[-1, 1]``.

    ``N = normal_transform_pixel3d(depth, height, width)`` returns a single matrix ``(1, 4, 4)`` that maps the
    homogeneous voxel coordinates ``(x, y, z, 1)`` of a ``depth`` by ``height`` by ``width`` volume to
    :math:`[-1, 1]`, corner-aligned, as :func:`normalize_homography3d` composes it. It has no ``align_corners``
    parameter, so a homography normalized with it suits only ``align_corners=True`` sampling, a restriction the 3D
    warps that use it inherit (`#4503 <https://github.com/kornia/kornia/issues/4503>`_). Its ``(x, y, z)`` order
    differs from the ``(d, x, y)`` of :func:`normalize_pixel_coordinates3d`.

    Convention:
        - the 3-D counterpart of
          :func:`~kornia.geometry.conversions.normal_transform_pixel`: same
          corner-aligned ``2 / (size - 1)`` scaling with offset ``-1``, same
          ``dtype=None`` / ``torch.get_default_dtype()`` rule, and likewise
          **never batched**. Singleton axes use the same invertible centre mapping, and zero or
          negative sizes raise ``ValueError``. Only the lines below differ
        - the result has shape ``(1, 4, 4)`` and acts on homogeneous
          ``(x, y, z, 1)`` column vectors with ``x`` scaled by ``width``, ``y``
          by ``height`` and ``z`` by ``depth``, while the positional argument
          order is ``(depth, height, width)``: ``normal_transform_pixel3d(9, 5, 3)``
          has diagonal ``[1.0, 0.5, 0.25]`` and sends ``(0, 0, 0)`` to
          ``(-1, -1, -1)`` and the far corner ``(2, 4, 8)`` to ``(1, 1, 1)``
        - that ``(x, y, z)`` slot order is a **permutation** of
          :func:`~kornia.geometry.conversions.normalize_pixel_coordinates3d`,
          which reads its input as ``(d, x, y)``. The two produce the same three
          numbers in different slots — with ``depth=9, height=5, width=3`` the
          point ``d=7, x=2, y=1`` gives ``[0.75, 1.0, -0.5]`` through the
          coordinate helper and ``[1.0, -0.5, 0.75]`` through this matrix — so a
          grid built for one silently permutes axes when fed to the other
        - an integer or bool ``dtype`` raises ``ValueError``, as in the 2-D
          function

    .. note::
        Legacy ``torch.jit.trace`` graphs assume positive runtime sizes for the
        same reason as the 2-D counterpart.

    .. deprecated:: 0.9.0
        ``eps`` is ignored and follows the deprecation policy of the 2-D
        counterpart.

    Args:
        depth: image depth.
        height: image height.
        width: image width.
        eps: deprecated compatibility parameter. It is ignored.
        device: device to place the result on.
        dtype: dtype of the result. ``None`` means ``torch.get_default_dtype()``.

    Returns:
        normalized transform with shape ``(1, 4, 4)``.

    See Also:
        - :func:`normal_transform_pixel`: the 2D ``(1, 3, 3)`` matrix, which also has the ``align_corners=False``
          map.
        - :func:`normalize_homography3d`: conjugates a voxel homography with this matrix.
        - :func:`normalize_pixel_coordinates3d`: the same scaling applied to ``(*, 3)`` points in ``(d, x, y)``
          order.

    Example:
        >>> normal_transform_pixel3d(2, 4, 5)
        tensor([[[ 0.5000,  0.0000,  0.0000, -1.0000],
                 [ 0.0000,  0.6667,  0.0000, -1.0000],
                 [ 0.0000,  0.0000,  2.0000, -1.0000],
                 [ 0.0000,  0.0000,  0.0000,  1.0000]]])

    """
    if not torch.jit.is_tracing() and (depth <= 0 or height <= 0 or width <= 0):
        raise ValueError(f"Input image size must be positive. Got depth={depth}, height={height}, width={width}.")
    # As in 2-D: an integer dtype truncates the 2/(size - 1) scales to 0 and returns a
    # rank-deficient matrix, so the dtype is rejected rather than silently truncated, and
    # uniformly rather than only for the sizes where the scale happens to be integral.
    # Raised directly, not through KORNIA_CHECK, which python -O and disable_checks()
    # switch off; the probe and its fast path are as in 2-D.
    if dtype is not None and dtype not in (
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.complex32,
        torch.complex64,
        torch.complex128,
    ):
        probe = torch.empty(0, dtype=dtype)
        if not (probe.is_floating_point() or probe.is_complex()):
            raise ValueError(f"dtype must be a floating point or complex type. Got {dtype}.")
    if not torch.jit.is_scripting() and not torch.jit.is_tracing() and not is_compiling() and eps != 1e-14:
        warnings.warn("`eps` is deprecated and ignored by `normal_transform_pixel3d`.", FutureWarning, stacklevel=2)

    if torch.jit.is_scripting() or (not torch.jit.is_tracing() and not is_compiling()):
        # As in 2-D, graph capture uses the tensor form below for symbolic sizes.
        sx = 1.0 if width == 1 else 2.0 / (width - 1.0)
        sy = 1.0 if height == 1 else 2.0 / (height - 1.0)
        sz = 1.0 if depth == 1 else 2.0 / (depth - 1.0)
        tx = 0.0 if width == 1 else -1.0
        ty = 0.0 if height == 1 else -1.0
        tz = 0.0 if depth == 1 else -1.0
        if torch.jit.is_scripting():
            # As in 2-D: a typed anchor keeps float64 constants exact under TorchScript.
            anchor = torch.ones((), device=device, dtype=dtype)
            zero = anchor * 0.0
            tr_mat = torch.stack(
                [
                    torch.stack([anchor * sx, zero, zero, anchor * tx]),
                    torch.stack([zero, anchor * sy, zero, anchor * ty]),
                    torch.stack([zero, zero, anchor * sz, anchor * tz]),
                    torch.stack([zero, zero, zero, anchor]),
                ]
            )
        elif dtype is None or dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            # As in 2-D: filled on the device, without a host copy that waits for the queue.
            tr_mat = _constant_tensor(
                [[sx, 0.0, 0.0, tx], [0.0, sy, 0.0, ty], [0.0, 0.0, sz, tz], [0.0, 0.0, 0.0, 1.0]],
                device=device,
                dtype=dtype,
            )
        else:
            # As in 2-D: complex and float8 dtypes keep the host literal.
            tr_mat = torch.tensor(
                [[sx, 0.0, 0.0, tx], [0.0, sy, 0.0, ty], [0.0, 0.0, sz, tz], [0.0, 0.0, 0.0, 1.0]],
                device=device,
                dtype=dtype,
            )
    else:
        # As in 2-D: keep the symbolic size arithmetic in at least float32, and resolve the
        # output dtype first so a ``dtype=None`` call under a half default dtype is promoted too.
        out_dtype = dtype if dtype is not None else torch.get_default_dtype()
        work_dtype = out_dtype if out_dtype in (torch.float32, torch.float64) else torch.float32
        width_t = torch.scalar_tensor(width, device=device, dtype=work_dtype)
        height_t = torch.scalar_tensor(height, device=device, dtype=work_dtype)
        depth_t = torch.scalar_tensor(depth, device=device, dtype=work_dtype)
        one = torch.ones((), device=device, dtype=work_dtype)
        zero = torch.zeros((), device=device, dtype=work_dtype)

        # As in 2-D, a singleton axis maps its only pixel to the normalized centre,
        # with unit scale so that homography composition can still invert the matrix.
        sx_t = torch.where(width_t == 1, one, 2.0 / (width_t - 1.0))
        sy_t = torch.where(height_t == 1, one, 2.0 / (height_t - 1.0))
        sz_t = torch.where(depth_t == 1, one, 2.0 / (depth_t - 1.0))
        tx_t = torch.where(width_t == 1, zero, -one)
        ty_t = torch.where(height_t == 1, zero, -one)
        tz_t = torch.where(depth_t == 1, zero, -one)

        tr_mat = torch.stack(
            [
                torch.stack([sx_t, zero, zero, tx_t]),
                torch.stack([zero, sy_t, zero, ty_t]),
                torch.stack([zero, zero, sz_t, tz_t]),
                torch.stack([zero, zero, zero, one]),
            ]
        ).to(dtype=out_dtype)  # 4x4

    return tr_mat.unsqueeze(0)  # 1x4x4


def _is_floating_or_complex(x: torch.Tensor) -> bool:
    """Whether ``x`` has a floating or complex dtype, the only dtypes in which a pixel normalization is invertible."""
    return x.is_floating_point() or x.is_complex()


def denormalize_homography(
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: tuple[int, int],
    dsize_dst: tuple[int, int],
    align_corners: bool = True,
) -> torch.Tensor:
    r"""Express a ``[-1, 1]`` homography ``(B, 3, 3)`` as a source→destination pixel homography.

    ``H = denormalize_homography(H_norm, dsize_src, dsize_dst)`` is the inverse of :func:`normalize_homography`: it
    takes homographies ``(B, 3, 3)``, or one ``(3, 3)``, from normalized :math:`[-1, 1]` source coordinates to
    normalized destination coordinates, and the ``(height, width)`` of both images, and returns the source→destination
    pixel homographies ``inv(N_dst) @ H_norm @ N_src`` ``(B, 3, 3)``. Pass the ``align_corners`` that produced
    ``H_norm``. For ``(B, 4, 4)`` voxel homographies, use :func:`denormalize_homography3d`.

    Convention:
        - the mirror of
          :func:`~kornia.geometry.conversions.normalize_homography`. The ``dsize``
          roles, the ``(x, y, 1)`` column vectors, per-sample batching, the
          ``align_corners``-selected frames and the shape guard are as
          documented there, including its integer-input warning
        - ``inv(N_dst)`` goes through ``torch.linalg.inv`` in eager mode (so
          ``cusolver`` on ``cuda``), not through the ``cross``-based adjugate
        - the two functions invert each other to a small multiple of the
          working dtype's ``eps`` times the largest entry, not bitwise in
          general

    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          denormalized. ``(B, 3, 3)``, or an unbatched ``(3, 3)``,
          which is promoted to ``(1, 3, 3)``
        dsize_src: size of the source image (height, width).
        dsize_dst: size of the destination image (height, width).
        align_corners: which :py:func:`torch.nn.functional.grid_sample` convention the
          incoming normalized coordinates follow. Must match the value used to produce
          them, so that this function inverts :func:`normalize_homography`.

    Returns:
        the denormalized homography of shape ``(B, 3, 3)``.

    See Also:
        - :func:`normalize_homography`: the forward conversion, pixels to normalized coordinates.
        - :func:`denormalize_homography3d`: the 3D form for ``(B, 4, 4)`` homographies, always corner-aligned.
        - :func:`normal_transform_pixel`: the ``(1, 3, 3)`` normalization matrix ``N`` composed here.

    """
    if not isinstance(dst_pix_trans_src_pix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(dst_pix_trans_src_pix)}")

    if not (dst_pix_trans_src_pix.ndim in (2, 3) and dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {dst_pix_trans_src_pix.shape}")

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    normalization_dtype = (
        dst_pix_trans_src_pix.dtype
        if dst_pix_trans_src_pix.is_floating_point() or dst_pix_trans_src_pix.is_complex()
        else None
    )
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(
        src_h,
        src_w,
        device=dst_pix_trans_src_pix.device,
        dtype=normalization_dtype,
        align_corners=align_corners,
    ).to(dst_pix_trans_src_pix)

    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(
        dst_h,
        dst_w,
        device=dst_pix_trans_src_pix.device,
        dtype=normalization_dtype,
        align_corners=align_corners,
    ).to(dst_pix_trans_src_pix)
    # A floating pixel normalization is triangular with diagonal scales 2 / (size - 1), so it is invertible unless a
    # scale underflows to 0, which in float16 takes an axis longer than 2**26 pixels; that matrix gives non-finite
    # values instead of an error. An integer one truncates its scales to 0 (kornia#3959) and keeps the error check.
    dst_denorm_trans_dst_pix = _torch_inverse_cast(
        dst_norm_trans_dst_pix, check_errors=not _is_floating_or_complex(dst_norm_trans_dst_pix)
    )
    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_denorm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_norm_trans_src_pix)
    return dst_norm_trans_src_norm


def normalize_homography3d(
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: tuple[int, int, int], dsize_dst: tuple[int, int, int]
) -> torch.Tensor:
    r"""Express a source→destination voxel homography ``(B, 4, 4)`` in ``[-1, 1]`` coordinates.

    ``H_norm = normalize_homography3d(H, dsize_src, dsize_dst)`` takes homographies ``(B, 4, 4)``, or one ``(4, 4)``,
    that map **source voxels to destination voxels**, and the ``(depth, height, width)`` of both volumes, and returns
    ``N_dst @ H @ inv(N_src)`` ``(B, 4, 4)`` with ``N`` from :func:`normal_transform_pixel3d`. It has no
    ``align_corners`` parameter: the result suits only ``align_corners=True`` sampling
    (`#4503 <https://github.com/kornia/kornia/issues/4503>`_). For the reverse, use :func:`denormalize_homography3d`;
    for ``(B, 3, 3)`` pixel homographies, :func:`normalize_homography`.

    Convention:
        - the 3-D counterpart of
          :func:`~kornia.geometry.conversions.normalize_homography`: same
          composition ``N_dst @ H @ inv(N_src)``, same source-to-destination
          direction re-expressed in normalized frames, same ``dsize_src`` on the
          right and ``dsize_dst`` on the left, and corner-aligned frames — with
          :func:`~kornia.geometry.conversions.normal_transform_pixel3d` in place
          of the 2-D helper. It is
          **not** the 2-D function with wider matrices, though: the shapes and
          — least visibly — the inversion routine differ, as the bullets below
          record
        - it inverts ``N_src`` with ``torch.linalg.inv`` (so ``cusolver`` on
          ``cuda``, and a ``float32`` upcast for ``float16``/``bfloat16``),
          where the 2-D function uses a closed-form adjugate, with the same
          ``float32`` upcast; the 2-D and 3-D routes therefore need not agree bitwise
        - the matrices are ``(B, 4, 4)`` and act on ``(x, y, z, 1)`` column
          vectors, while both ``dsize`` arguments are
          ``(depth, height, width)`` triples
        - :func:`~kornia.geometry.conversions.denormalize_homography3d`
          computes the reverse ``inv(N_dst) @ H @ N_src``
        - the shape guard is the 2-D function's for 4x4 matrices: ``(4, 4)``
          (returned as ``(1, 4, 4)``) or ``(B, 4, 4)``, anything else raises
          ``ValueError``

    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          normalized. ``(B, 4, 4)``, or an unbatched ``(4, 4)``, which
          is promoted to ``(1, 4, 4)``
        dsize_src: size of the source image (depth, height, width).
        dsize_dst: size of the destination image (depth, height, width).

    Returns:
        the normalized homography.

    Shape:
        Output: ``(B, 4, 4)``

    See Also:
        - :func:`denormalize_homography3d`: the reverse, a normalized voxel homography back to voxels.
        - :func:`normalize_homography`: the 2D form for ``(B, 3, 3)`` pixel homographies, with ``align_corners``.
        - :func:`normal_transform_pixel3d`: the ``(1, 4, 4)`` normalization matrix ``N`` composed here.

    """
    if not isinstance(dst_pix_trans_src_pix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(dst_pix_trans_src_pix)}")

    if not (dst_pix_trans_src_pix.ndim in (2, 3) and dst_pix_trans_src_pix.shape[-2:] == (4, 4)):
        raise ValueError(f"Input dst_pix_trans_src_pix must be a Bx4x4 tensor. Got {dst_pix_trans_src_pix.shape}")

    # source and destination sizes
    src_d, src_h, src_w = dsize_src
    dst_d, dst_h, dst_w = dsize_dst
    # compute the transformation pixel/norm for src/dst
    normalization_dtype = (
        dst_pix_trans_src_pix.dtype
        if dst_pix_trans_src_pix.is_floating_point() or dst_pix_trans_src_pix.is_complex()
        else None
    )
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel3d(
        src_d, src_h, src_w, device=dst_pix_trans_src_pix.device, dtype=normalization_dtype
    ).to(dst_pix_trans_src_pix)

    # A floating pixel normalization is invertible unless a scale underflows (see `denormalize_homography`); an
    # integer one keeps the error check (kornia#3959).
    src_pix_trans_src_norm = _torch_inverse_cast(
        src_norm_trans_src_pix, check_errors=not _is_floating_or_complex(src_norm_trans_src_pix)
    )
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel3d(
        dst_d, dst_h, dst_w, device=dst_pix_trans_src_pix.device, dtype=normalization_dtype
    ).to(dst_pix_trans_src_pix)
    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def denormalize_homography3d(
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: tuple[int, int, int], dsize_dst: tuple[int, int, int]
) -> torch.Tensor:
    r"""Express a ``[-1, 1]`` homography ``(B, 4, 4)`` as a source→destination voxel homography.

    ``H = denormalize_homography3d(H_norm, dsize_src, dsize_dst)`` is the inverse of :func:`normalize_homography3d`: it
    takes real floating homographies ``(B, 4, 4)``, or one ``(4, 4)``, from normalized :math:`[-1, 1]` source
    coordinates to normalized destination coordinates, and the ``(depth, height, width)`` of both volumes, and returns
    the source→destination voxel homographies ``inv(N_dst) @ H_norm @ N_src`` ``(B, 4, 4)``, always corner-aligned
    (`#4503 <https://github.com/kornia/kornia/issues/4503>`_). For ``(B, 3, 3)`` pixel homographies, use
    :func:`denormalize_homography`.

    Convention:
        - the mirror of
          :func:`~kornia.geometry.conversions.normalize_homography3d`. The ``(depth, height, width)``
          ``dsize`` roles, the ``(x, y, z, 1)`` column vectors, per-sample
          batching, the corner-aligned frames and the shape guard are as
          documented there
        - ``inv(N_dst)`` goes through ``torch.linalg.inv`` (so ``cusolver`` on
          ``cuda``, and a ``float32`` upcast for ``float16``/``bfloat16``), the
          same routine :func:`~kornia.geometry.conversions.normalize_homography3d`
          uses for ``inv(N_src)``
        - the two functions invert each other to a small multiple of the
          working dtype's ``eps`` times the largest entry, not bitwise in
          general
        - only real floating point homographies are accepted (``float16``,
          ``bfloat16``, ``float32``, ``float64``): an integer, bool or complex
          ``dst_pix_trans_src_pix`` raises ``TypeCheckError`` (also a
          ``TypeError``) instead of being promoted, so cast it first. Integers
          would truncate the normalization matrices (the 2-D function still
          does), and
          complex dtypes would be inverted in ``float32``
          (``_torch_inverse_cast``), losing ``complex128`` precision

    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          denormalized. ``(B, 4, 4)``, or an unbatched ``(4, 4)``, which
          is promoted to ``(1, 4, 4)``
        dsize_src: size of the source image (depth, height, width).
        dsize_dst: size of the destination image (depth, height, width).

    Returns:
        the denormalized homography.

    Shape:
        Output: ``(B, 4, 4)``

    See Also:
        - :func:`normalize_homography3d`: the forward conversion, voxels to normalized coordinates.
        - :func:`denormalize_homography`: the 2D form for ``(B, 3, 3)`` homographies, with ``align_corners``.
        - :func:`normal_transform_pixel3d`: the ``(1, 4, 4)`` normalization matrix ``N`` composed here.

    """
    if not isinstance(dst_pix_trans_src_pix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(dst_pix_trans_src_pix)}")

    # An integer or bool homography would cast the normalization matrices back to integers
    # (truncating the 2/(size - 1) scales to 0, or failing on a singular inverse), and a complex one
    # would be inverted in float32 by _torch_inverse_cast, so the dtype is rejected rather than
    # promoted: the caller casts. Raised directly, not through KORNIA_CHECK,
    # which python -O and disable_checks() switch off.
    if not dst_pix_trans_src_pix.is_floating_point():
        raise TypeCheckError(
            "Input dst_pix_trans_src_pix must be a real floating point tensor. "
            f"Got dtype {dst_pix_trans_src_pix.dtype}.",
            actual_type=dst_pix_trans_src_pix.dtype,
        )

    if not (dst_pix_trans_src_pix.ndim in (2, 3) and dst_pix_trans_src_pix.shape[-2:] == (4, 4)):
        raise ValueError(f"Input dst_pix_trans_src_pix must be a Bx4x4 tensor. Got {dst_pix_trans_src_pix.shape}")

    # source and destination sizes
    src_d, src_h, src_w = dsize_src
    dst_d, dst_h, dst_w = dsize_dst
    # compute the transformation pixel/norm for src/dst
    normalization_dtype = dst_pix_trans_src_pix.dtype
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel3d(
        src_d, src_h, src_w, device=dst_pix_trans_src_pix.device, dtype=normalization_dtype
    ).to(dst_pix_trans_src_pix)

    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel3d(
        dst_d, dst_h, dst_w, device=dst_pix_trans_src_pix.device, dtype=normalization_dtype
    ).to(dst_pix_trans_src_pix)
    # A floating pixel normalization is invertible unless a scale underflows (see `denormalize_homography`); an
    # integer one keeps the error check (kornia#3959).
    dst_pix_trans_dst_norm = _torch_inverse_cast(
        dst_norm_trans_dst_pix, check_errors=not _is_floating_or_complex(dst_norm_trans_dst_pix)
    )
    # compute chain transformations
    return dst_pix_trans_dst_norm @ (dst_pix_trans_src_pix @ src_norm_trans_src_pix)


def normalize_points_with_intrinsics(point_2d: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
    """Convert pixel coordinates (u, v) to normalized camera coordinates (x, y) with the intrinsics K.

    ``xy = normalize_points_with_intrinsics(uv, K)`` takes pixel points ``(*, 2)`` and pinhole intrinsics
    ``(*, 3, 3)`` and returns the normalized camera coordinates ``((u - cx) / fx, (v - cy) / fy)``, ``(*, 2)``: the
    point where the ray of each pixel meets the ``z = 1`` plane, the form an essential matrix relates. A
    ``(B, 3, 3)`` ``K`` pairs camera ``b`` with ``uv[b]`` for ``(B, 2)`` and ``(B, N, 2)`` points; for
    ``(B, H, W, 2)`` points pass ``K[:, None, None]``, since the cameras otherwise line up with the ``H`` axis
    (`#6409 <https://github.com/kornia/kornia/issues/6409>`_). For the reverse, use
    :func:`denormalize_points_with_intrinsics`; for 3D points at a depth,
    :func:`~kornia.geometry.camera.perspective.unproject_points`.

    Convention:
        See :doc:`camera and world conventions </get-started/camera-conventions>` for the camera intrinsics and
        pixel-centre conventions used by these coordinates.

        - ``point_2d`` is ``(*, 2)`` in pixel ``(u, v)`` order and
          ``camera_matrix`` is a row-major pinhole ``(*, 3, 3)`` with
          ``fx = K[0, 0]``, ``fy = K[1, 1]``, ``cx = K[0, 2]``, ``cy = K[1, 2]``
        - with ``K = [[100., 0., 320.], [0., 200., 240.], [0., 0., 1.]]`` the
          pixel ``(420., 440.)`` maps to ``(1., 1.)``
        - the skew entry ``K[0, 1]`` is **ignored**; only the two diagonal
          entries and the ``[:2, 2]`` column are read. This is a design choice:
          kornia's functional pinhole model is zero-skew, like OpenCV's, and
          :func:`~kornia.geometry.camera.perspective.project_points` reads
          ``K`` the same way. For a skewed ``K`` apply ``inv(K)`` directly

    Args:
        point_2d: tensor containing the 2d points in the image pixel coordinates. The shape of the tensor can be
                  ``(*, 2)``.
        camera_matrix: tensor containing the intrinsics camera matrix. The tensor shape must be ``(*, 3, 3)``.

    Returns:
        tensor of normalized camera coordinates :math:`(x, y)` with shape ``(*, 2)``, the leading dimensions
        broadcast from both inputs.

    See Also:
        - :func:`denormalize_points_with_intrinsics`: the reverse, normalized camera coordinates back to pixels.
        - :func:`~kornia.geometry.camera.perspective.unproject_points`: the 3D camera-frame points of pixels at a
          given depth.
        - :func:`normalize_pixel_coordinates`: a different normalization, to :math:`[-1, 1]` by the image size.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> X = torch.rand(1, 2)
        >>> K = torch.eye(3)[None]
        >>> normalize_points_with_intrinsics(X, K)
        tensor([[0.4963, 0.7682]])

    """
    KORNIA_CHECK_SHAPE(point_2d, ["*", "2"])
    KORNIA_CHECK_SHAPE(camera_matrix, ["*", "3", "3"])
    # projection eq. K_inv * [u v 1]'
    # x = (u - cx) * Z / fx
    # y = (v - cy) * Z / fy

    # unpack coordinates
    cxcy = camera_matrix[..., :2, 2]
    fxfy = camera_matrix[..., :2, :2].diagonal(dim1=-2, dim2=-1)
    if len(cxcy.shape) < len(point_2d.shape):  # broadcast intrinsics:
        cxcy, fxfy = cxcy.unsqueeze(-2), fxfy.unsqueeze(-2)
    return (point_2d - cxcy) / fxfy


def denormalize_points_with_intrinsics(point_2d_norm: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
    """Convert normalized camera coordinates (x, y) to pixel coordinates (u, v) with the intrinsics K.

    ``uv = denormalize_points_with_intrinsics(xy, K)`` is the inverse of :func:`normalize_points_with_intrinsics`: it
    takes normalized camera coordinates ``(*, 2)``, points on the ``z = 1`` plane, and pinhole intrinsics
    ``(*, 3, 3)``, and returns the pixels ``(x * fx + cx, y * fy + cy)``, ``(*, 2)``. ``K`` broadcasts against the
    points as in that function: for ``(B, H, W, 2)`` points pass ``K[:, None, None]``
    (`#6409 <https://github.com/kornia/kornia/issues/6409>`_). To project 3D camera-frame points instead, use
    :func:`~kornia.geometry.camera.perspective.project_points`.

    Convention:
        See :doc:`camera and world conventions </get-started/camera-conventions>` for the camera intrinsics and
        pixel-centre conventions used by these coordinates.

        - the ``K`` layout is documented on
          :func:`~kornia.geometry.conversions.normalize_points_with_intrinsics`,
          and the skew entry ``K[0, 1]`` is ignored here too

    Args:
        point_2d_norm: tensor containing the 2d points in normalized camera coordinates. The shape of the tensor can
                       be ``(*, 2)``.
        camera_matrix: tensor containing the intrinsics camera matrix. The tensor shape must be ``(*, 3, 3)``.

    Returns:
        tensor of :math:`(u, v)` pixel coordinates with shape ``(*, 2)``, the leading dimensions broadcast from both
        inputs.

    See Also:
        - :func:`normalize_points_with_intrinsics`: the forward conversion, pixels to normalized camera coordinates.
        - :func:`~kornia.geometry.camera.perspective.project_points`: the pixels of 3D camera-frame points.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> X = torch.rand(1, 2)
        >>> K = torch.eye(3)[None]
        >>> denormalize_points_with_intrinsics(X, K)
        tensor([[0.4963, 0.7682]])

    """
    KORNIA_CHECK_SHAPE(point_2d_norm, ["*", "2"])
    KORNIA_CHECK_SHAPE(camera_matrix, ["*", "3", "3"])
    # projection eq. [u, v, w]' = K * [x y z 1]'
    # u = fx * X + cx
    # v = fy * Y + cy

    fxfy = camera_matrix[..., :2, :2].diagonal(dim1=-2, dim2=-1)  # (*, 2)
    cxcy = camera_matrix[..., :2, 2]  # (*, 2)
    if len(cxcy.shape) < len(point_2d_norm.shape):
        fxfy, cxcy = fxfy.unsqueeze(-2), cxcy.unsqueeze(-2)
    return point_2d_norm * fxfy + cxcy


def Rt_to_matrix4x4(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""Pack rotations ``(B, 3, 3)`` and translations ``(B, 3, 1)`` into ``(B, 4, 4)`` pose matrices.

    ``M = Rt_to_matrix4x4(R, t)`` places each rotation ``(B, 3, 3)`` in the top-left block and its translation
    ``(B, 3, 1)`` in the last column of a ``(B, 4, 4)`` matrix with the bottom row ``[0, 0, 0, 1]``, so that ``M``
    maps ``x`` to ``R @ x + t``. It checks neither that ``R`` is a rotation nor which way the pose goes:
    camera-to-world and world-to-camera poses pack the same way. For the reverse, use :func:`matrix4x4_to_Rt`; for the
    ``(*, 3, 4)`` matrix ``[R | t]`` without the bottom row,
    :func:`~kornia.geometry.transform.projection_from_Rt`.

    Convention:
        - ``t`` goes in the **last column** and the appended bottom row is
          ``[0, 0, 0, 1]``, so the result acts on homogeneous column vectors as
          ``x_out = R @ x_in + t``: with ``t = (1, 2, 3)``,
          ``M @ (0, 0, 0, 1)`` returns ``t`` itself and ``M @ (1, 0, 0, 1)``
          returns ``R[:, 0] + t``
        - the function is **frame-agnostic** — it packs whatever ``(R, t)`` it
          is handed and does not check that ``R`` is a rotation. Under the
          camera-to-world reading the rest of this family uses (see
          :func:`~kornia.geometry.conversions.camtoworld_to_worldtocam_Rt`),
          ``t`` is the **camera centre in world coordinates**, whereas the
          world-to-camera form's ``-R^T t`` is a translation and not a centre.
          Which of the two you are packing is your bookkeeping, not this
          function's
        - shapes are strict: exactly ``(B, 3, 3)`` and ``(B, 3, 1)``.
          An unbatched ``(3, 3)``, a ``(B, 3)`` translation, a ``(B, 1, 3)``
          translation and extra leading dimensions each raise ``ShapeError``;
          ``R`` and ``t`` must also carry the same batch size: a mismatched
          pair raises ``ShapeError`` too
        - :func:`~kornia.geometry.conversions.matrix4x4_to_Rt` is the inverse:
          ``Rt -> 4x4 -> Rt`` is bitwise, while ``4x4 -> Rt -> 4x4`` is bitwise
          only when the bottom row is already ``[0, 0, 0, 1]``, because
          :func:`~kornia.geometry.conversions.matrix4x4_to_Rt` drops it

    Note:
        ``R`` and ``t`` must promote to a floating point or complex dtype;
        an integer or boolean pair raises ``ValueError`` naming its dtype.
        Mixed integer/floating inputs keep torch's promotion. The ``_4x4``
        frame functions and ``camtoworld_to_worldtocam_Rt`` only transpose,
        negate and multiply, so they accept integer input as well.

    Args:
        R: Rotation matrix, ``(B, 3, 3)``.
        t: Translation matrix ``(B, 3, 1)``.

    Returns:
        the extrinsics ``(B, 4, 4)``.

    See Also:
        - :func:`matrix4x4_to_Rt`: the reverse, splitting a ``(B, 4, 4)`` matrix into ``R`` and ``t``.
        - :func:`camtoworld_to_worldtocam_Rt`: inverts a rigid ``(R, t)`` pose before or after packing.
        - :func:`~kornia.geometry.transform.projection_from_Rt`: the ``(*, 3, 4)`` matrix ``[R | t]``, without the
          bottom row.
        - :func:`convert_affinematrix_to_homography3d`: the same ``(B, 4, 4)`` from a ``(B, 3, 4)`` matrix.

    Example:
        >>> R, t = torch.eye(3)[None], torch.ones(3).reshape(1, 3, 1)
        >>> Rt_to_matrix4x4(R, t)
        tensor([[[1., 0., 0., 1.],
                 [0., 1., 0., 1.],
                 [0., 0., 1., 1.],
                 [0., 0., 0., 1.]]])

    """
    KORNIA_CHECK_SHAPE(R, ["B", "3", "3"])
    KORNIA_CHECK_SHAPE(t, ["B", "3", "1"])
    _check_Rt_same_batch(R, t, "Rt_to_matrix4x4")
    Rt = torch.cat([R, t], dim=2)
    if not Rt.is_floating_point() and not Rt.is_complex():
        raise ValueError(f"R and t must promote to a floating point or complex dtype. Got {Rt.dtype}.")
    return convert_affinematrix_to_homography3d(Rt)


def matrix4x4_to_Rt(extrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Split ``(B, 4, 4)`` pose matrices into rotation ``(B, 3, 3)`` and translation ``(B, 3, 1)`` views.

    ``R, t = matrix4x4_to_Rt(M)`` returns the top-left ``(B, 3, 3)`` block and the first three rows ``(B, 3, 1)`` of
    the last column of each ``(B, 4, 4)`` matrix, as views of ``M``; the bottom row is ignored. It is the inverse of
    :func:`Rt_to_matrix4x4`.

    Convention:
        - the inverse of
          :func:`~kornia.geometry.conversions.Rt_to_matrix4x4`, with the same
          layout and the same frame-agnosticism. Only the lines below differ
        - the input shape is strictly ``(B, 4, 4)``; an unbatched ``(4, 4)``,
          a ``(B, 3, 3)`` and extra leading dimensions each raise ``ShapeError``
        - the **bottom row is ignored**: a projective 4x4 is silently truncated
          to its rigid part, so replacing the bottom row with ``[9, 9, 9, 9]``
          leaves ``R`` and ``t`` bit-identical, and re-packing through
          :func:`~kornia.geometry.conversions.Rt_to_matrix4x4` restores the
          canonical ``[0, 0, 0, 1]`` rather than what was passed in
        - the returned ``R`` and ``t`` are **views** of ``extrinsics``, not
          copies: ``R.mul_(0.)`` zeroes the caller's rotation block in place.
          Clone before mutating either output

    Args:
        extrinsics: pose matrix ``(B, 4, 4)``.

    Returns:
        R: Rotation matrix, ``(B, 3, 3)``.
        t: Translation matrix ``(B, 3, 1)``.

    See Also:
        - :func:`Rt_to_matrix4x4`: the reverse, packing ``R`` and ``t`` with the bottom row ``[0, 0, 0, 1]``.
        - :func:`~kornia.geometry.linalg.inverse_transformation`: the rigid inverse of the ``(B, 4, 4)`` matrix
          itself.

    Example:
        >>> ext = torch.eye(4)[None]
        >>> matrix4x4_to_Rt(ext)
        (tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]]), tensor([[[0.],
                 [0.],
                 [0.]]]))

    """
    KORNIA_CHECK_SHAPE(extrinsics, ["B", "4", "4"])
    R, t = extrinsics[:, :3, :3], extrinsics[:, :3, 3:]
    return R, t


def _integer_safe_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # ``a @ b`` for the frame and pose functions that accept integer input (see the Rt_to_matrix4x4 Note).
    # CUDA has no integer matmul kernels ("baddbmm_cuda" not implemented for 'Long'), so a pair of one integer
    # dtype is multiplied out and summed instead, on every device so that CPU covers this branch too. Integer
    # products and sums wrap the same way in either form, so the result is bitwise CPU's integer ``a @ b``.
    # Every other pair (floating, complex, bool, the unsigned dtypes CPU matmul lacks, mixed dtypes) keeps
    # ``a @ b``, its rounding and its errors. So does a pair whose inner sizes differ, which only reaches here
    # with checks disabled: broadcasting would stretch a size-1 inner dimension where ``a @ b`` raises.
    if (
        a.dtype == b.dtype
        and a.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
        and a.dim() > 1
        and b.dim() > 1
        and a.shape[-1] == b.shape[-2]
    ):
        return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2, dtype=a.dtype)
    return a @ b


def camtoworld_graphics_to_vision_4x4(extrinsics_graphics: torch.Tensor) -> torch.Tensor:
    r"""Convert ``(B, 4, 4)`` camera-to-world poses from graphics (OpenGL) to vision (OpenCV) camera axes.

    ``M_vision = camtoworld_graphics_to_vision_4x4(M_graphics)`` takes camera-to-world poses ``(B, 4, 4)`` whose
    camera axes follow the graphics convention, ``[+x, +y, +z] == [right, up, backwards]``, and returns them with the
    vision camera axes, ``[right, down, forwards]``: it negates the camera ``y`` and ``z`` axes, the second and third
    columns of the rotation block, and leaves the camera centre unchanged. A world-to-camera pose needs the flip on the
    other side, which this function does not make: invert it to camera-to-world with
    :func:`~kornia.geometry.linalg.inverse_transformation`, convert, and invert the result back. For a pose given as
    ``(R, t)``, use :func:`camtoworld_graphics_to_vision_Rt`.

    Convention:
        See :doc:`camera and world conventions </get-started/camera-conventions>` for the graphics/vision camera
        frames and camera-to-world extrinsics convention.

        - the input is a **camera-to-world** pose ``(B, 4, 4)``: its 3x3
          block maps camera axes into the world and its last column is the
          camera centre in world coordinates
        - the operation is exactly ``extrinsics @ diag(1, -1, -1, 1)`` — a
          **right** multiplication, i.e. a change of the **camera-side** basis.
          The translation column is untouched. Left-multiplying by the same
          matrix would negate two of its components instead, and feeding a
          *world-to-camera* matrix in flips the wrong side and leaves the
          translation in the old frame: a silent error, not an exception
        - graphics (OpenGL) camera frame: ``+x`` right, ``+y`` up, ``+z``
          backwards — the camera looks down ``-z``. Vision (OpenCV) camera
          frame: ``+x`` right, ``+y`` down, ``+z`` forwards — the camera looks
          down ``+z``. On the identity camera-to-world pose, camera ``+y``
          maps to world ``+y`` before the conversion and to world ``-y`` after
          it, and likewise for ``+z``
        - the flip **preserves the determinant** of the 3x3 block, and with it
          the handedness: ``diag(1, -1, -1)`` has determinant ``+1``, so a
          proper rotation stays proper. Nothing here checks that the input is a
          rotation
        - the flip is an **involution**, and all four graphics/vision frame
          functions compute the identical map:
          :func:`~kornia.geometry.conversions.camtoworld_vision_to_graphics_4x4`
          returns the same matrix as this one on the same input, and applying
          either twice returns the input. The two names document the caller's
          intent, not different arithmetic
        - the shape is strictly ``(B, 4, 4)``; the ``_Rt`` variants take and
          return ``(B, 3, 3)`` and ``(B, 3, 1)`` and agree with this
          path bitwise

    .. warning::
        Integer input is not guarded: this function returns ``int64`` for an
        ``int64`` matrix on every device, CUDA (which has no integer matmul)
        included, while
        :func:`~kornia.geometry.conversions.camtoworld_graphics_to_vision_Rt`
        raises ``ValueError`` naming the unsupported dtype; see
        :func:`~kornia.geometry.conversions.Rt_to_matrix4x4`.

    Args:
        extrinsics_graphics: pose matrix ``(B, 4, 4)``.

    Returns:
        extrinsics: pose matrix ``(B, 4, 4)``.

    See Also:
        - :func:`camtoworld_vision_to_graphics_4x4`: the reverse direction, the same arithmetic.
        - :func:`camtoworld_graphics_to_vision_Rt`: the same conversion for a pose given as ``(R, t)``.
        - :func:`camtoworld_to_worldtocam_Rt`: inverts a camera-to-world pose to world-to-camera.

    Example:
        >>> ext = torch.eye(4)[None]
        >>> camtoworld_graphics_to_vision_4x4(ext)
        tensor([[[ 1.,  0.,  0.,  0.],
                 [ 0., -1.,  0.,  0.],
                 [ 0.,  0., -1.,  0.],
                 [ 0.,  0.,  0.,  1.]]])

    """
    KORNIA_CHECK_SHAPE(extrinsics_graphics, ["B", "4", "4"])
    invert_yz = torch.tensor(
        [[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.0]]],
        dtype=extrinsics_graphics.dtype,
        device=extrinsics_graphics.device,
    )
    return _integer_safe_matmul(extrinsics_graphics, invert_yz)


def camtoworld_graphics_to_vision_Rt(R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert camera-to-world ``(R, t)`` poses from graphics (OpenGL) to vision (OpenCV) camera axes.

    ``R_vision, t = camtoworld_graphics_to_vision_Rt(R, t)`` is :func:`camtoworld_graphics_to_vision_4x4` for a
    camera-to-world pose given as rotations ``(B, 3, 3)`` and camera centres ``(B, 3, 1)``: it negates the camera
    ``y`` and ``z`` axes, the second and third columns of ``R``, to turn graphics camera axes
    ``[right, up, backwards]`` into vision ones ``[right, down, forwards]``, and returns ``t`` unchanged.

    Convention:
        See :doc:`camera and world conventions </get-started/camera-conventions>` for the graphics/vision camera
        frames and camera-to-world extrinsics convention.

        - the split-argument form of
          :func:`~kornia.geometry.conversions.camtoworld_graphics_to_vision_4x4`,
          which documents the flip, the two camera frames, the involution and
          the preserved handedness. It packs ``(R, t)`` with
          :func:`~kornia.geometry.conversions.Rt_to_matrix4x4`, applies that
          function and splits the result again, so the two paths agree bitwise.
          Only the lines below differ
        - the shapes are strictly ``(B, 3, 3)`` and ``(B, 3, 1)`` in and
          out, with the same-batch-size rule and the integer-input ``ValueError`` of
          :func:`~kornia.geometry.conversions.Rt_to_matrix4x4`

    Args:
        R: Rotation matrix, ``(B, 3, 3)``.
        t: Translation matrix ``(B, 3, 1)``.

    Returns:
        R: Rotation matrix, ``(B, 3, 3)``.
        t: Translation matrix ``(B, 3, 1)``.

    See Also:
        - :func:`camtoworld_vision_to_graphics_Rt`: the reverse direction, the same arithmetic.
        - :func:`camtoworld_graphics_to_vision_4x4`: the same conversion for a ``(B, 4, 4)`` matrix.
        - :func:`camtoworld_to_worldtocam_Rt`: inverts a camera-to-world ``(R, t)`` to world-to-camera.

    Example:
        >>> R, t = torch.eye(3)[None], torch.ones(3).reshape(1, 3, 1)
        >>> camtoworld_graphics_to_vision_Rt(R, t)
        (tensor([[[ 1.,  0.,  0.],
                 [ 0., -1.,  0.],
                 [ 0.,  0., -1.]]]), tensor([[[1.],
                 [1.],
                 [1.]]]))

    """
    KORNIA_CHECK_SHAPE(R, ["B", "3", "3"])
    KORNIA_CHECK_SHAPE(t, ["B", "3", "1"])
    _check_Rt_same_batch(R, t, "camtoworld_graphics_to_vision_Rt")
    mat4x4 = camtoworld_graphics_to_vision_4x4(Rt_to_matrix4x4(R, t))
    return matrix4x4_to_Rt(mat4x4)


def camtoworld_vision_to_graphics_4x4(extrinsics_vision: torch.Tensor) -> torch.Tensor:
    r"""Convert ``(B, 4, 4)`` camera-to-world poses from vision (OpenCV) to graphics (OpenGL) camera axes.

    ``M_graphics = camtoworld_vision_to_graphics_4x4(M_vision)`` takes camera-to-world poses ``(B, 4, 4)`` whose
    camera axes follow the vision convention, ``[+x, +y, +z] == [right, down, forwards]``, and returns them with the
    graphics camera axes, ``[right, up, backwards]``, by negating the camera ``y`` and ``z`` axes. The flip is its own
    inverse, so this is the same arithmetic as :func:`camtoworld_graphics_to_vision_4x4`; the name records the
    direction. For a pose given as ``(R, t)``, use :func:`camtoworld_vision_to_graphics_Rt`.

    Convention:
        See :doc:`camera and world conventions </get-started/camera-conventions>` for the graphics/vision camera
        frames and camera-to-world extrinsics convention.

        - the same map as
          :func:`~kornia.geometry.conversions.camtoworld_graphics_to_vision_4x4`,
          which carries the canonical block: ``diag(1, -1, -1, 1)`` is its own
          inverse, so vision-to-graphics and graphics-to-vision are the identical
          function and return bitwise equal matrices on the same input. The name
          records which direction the caller means
        - everything else — the right multiplication, the untouched translation
          column, the two frame definitions, the preserved determinant, the
          strict ``(B, 4, 4)`` shape and the integer-dtype warning — is as
          documented there

    Args:
        extrinsics_vision: pose matrix ``(B, 4, 4)``.

    Returns:
        extrinsics: pose matrix ``(B, 4, 4)``.

    See Also:
        - :func:`camtoworld_graphics_to_vision_4x4`: the forward direction, which documents the flip.
        - :func:`camtoworld_vision_to_graphics_Rt`: the same conversion for a pose given as ``(R, t)``.

    Example:
        >>> ext = torch.eye(4)[None]
        >>> camtoworld_vision_to_graphics_4x4(ext)
        tensor([[[ 1.,  0.,  0.,  0.],
                 [ 0., -1.,  0.,  0.],
                 [ 0.,  0., -1.,  0.],
                 [ 0.,  0.,  0.,  1.]]])

    """
    KORNIA_CHECK_SHAPE(extrinsics_vision, ["B", "4", "4"])
    invert_yz = torch.tensor(
        [[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.0]]],
        dtype=extrinsics_vision.dtype,
        device=extrinsics_vision.device,
    )
    return _integer_safe_matmul(extrinsics_vision, invert_yz)


def camtoworld_vision_to_graphics_Rt(R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert camera-to-world ``(R, t)`` poses from vision (OpenCV) to graphics (OpenGL) camera axes.

    ``R_graphics, t = camtoworld_vision_to_graphics_Rt(R, t)`` is :func:`camtoworld_vision_to_graphics_4x4` for a
    camera-to-world pose given as rotations ``(B, 3, 3)`` and camera centres ``(B, 3, 1)``: it negates the second and
    third columns of ``R`` and returns ``t`` unchanged, the same arithmetic as
    :func:`camtoworld_graphics_to_vision_Rt`.

    Convention:
        See :doc:`camera and world conventions </get-started/camera-conventions>` for the graphics/vision camera
        frames and camera-to-world extrinsics convention.

        - the split-argument form of
          :func:`~kornia.geometry.conversions.camtoworld_vision_to_graphics_4x4`
          and, because ``diag(1, -1, -1, 1)`` is its own inverse, bitwise the
          same function as
          :func:`~kornia.geometry.conversions.camtoworld_graphics_to_vision_Rt`.
          The canonical block lives on
          :func:`~kornia.geometry.conversions.camtoworld_graphics_to_vision_4x4`
        - the shapes, the unchanged ``t`` and the warnings are as in the
          graphics-to-vision ``_Rt`` form

    Args:
        R: Rotation matrix, ``(B, 3, 3)``.
        t: Translation matrix ``(B, 3, 1)``.

    Returns:
        R: Rotation matrix, ``(B, 3, 3)``.
        t: Translation matrix ``(B, 3, 1)``.

    See Also:
        - :func:`camtoworld_graphics_to_vision_Rt`: the forward direction, the same arithmetic.
        - :func:`camtoworld_vision_to_graphics_4x4`: the same conversion for a ``(B, 4, 4)`` matrix.

    Example:
        >>> R, t = torch.eye(3)[None], torch.ones(3).reshape(1, 3, 1)
        >>> camtoworld_vision_to_graphics_Rt(R, t)
        (tensor([[[ 1.,  0.,  0.],
                 [ 0., -1.,  0.],
                 [ 0.,  0., -1.]]]), tensor([[[1.],
                 [1.],
                 [1.]]]))

    """
    KORNIA_CHECK_SHAPE(R, ["B", "3", "3"])
    KORNIA_CHECK_SHAPE(t, ["B", "3", "1"])
    _check_Rt_same_batch(R, t, "camtoworld_vision_to_graphics_Rt")
    mat4x4 = camtoworld_vision_to_graphics_4x4(Rt_to_matrix4x4(R, t))
    return matrix4x4_to_Rt(mat4x4)


def _check_Rt_same_batch(R: torch.Tensor, t: torch.Tensor, fn_name: str) -> None:
    # KORNIA_CHECK_SHAPE validates each argument on its own; this guard compares the two batch
    # sizes, so a mismatched (R, t) pair raises here with both shapes in the message. Like the
    # KORNIA_CHECK helpers it follows disable_checks(): read the flag at call time, not a copy
    # bound at import.
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not are_checks_enabled():
            return

    if R.shape[0] != t.shape[0]:
        raise ShapeError(
            f"{fn_name}: R and t must have the same batch size, got {list(R.shape)} and {list(t.shape)}.",
            actual_shape=list(t.shape),
            expected_shape=[str(R.shape[0]), "3", "1"],
        )


def _check_is_rotation(R: torch.Tensor, fn_name: str) -> None:
    # Opt-in validation behind check_rotation=True; the default path stays unchecked.
    # det has no half or integer kernels, so the check runs in float32 unless R is already float64.
    # The tolerance follows R's own precision: 100 eps, but 16 eps for float16/bfloat16, where 100 eps
    # (0.78 in bfloat16) would accept a 50% shear. Every tolerance stays below 1/3, so an R that passes
    # the orthogonality test is non-singular and det(R) <= 0 can only mean a reflection.
    # eps literals rather than torch.finfo, which TorchScript cannot compile.
    work = R if R.dtype == torch.float64 else R.to(torch.float32)
    if R.dtype == torch.float64:
        tol = 100.0 * 2.0**-52
    elif R.dtype == torch.bfloat16:
        tol = 16.0 * 2.0**-7
    elif R.dtype == torch.float16:
        tol = 16.0 * 2.0**-10
    else:  # float32, and integer input checked in float32
        tol = 100.0 * 2.0**-23
    eye = torch.eye(3, device=R.device, dtype=work.dtype)
    ortho_err = (work @ work.transpose(-2, -1) - eye).abs().amax(dim=(-2, -1))
    # "not all within" rather than "any above", so that a NaN entry fails the check.
    if not bool((ortho_err <= tol).all()):
        raise ValueError(
            f"{fn_name}: R is not a rotation matrix: max|R @ R^T - I| = {float(ortho_err.max())} "
            f"is not within the tolerance {tol}."
        )
    if bool((torch.linalg.det(work) <= 0).any()):
        raise ValueError(f"{fn_name}: R is a reflection (det(R) < 0), not a rotation matrix.")


def camtoworld_to_worldtocam_Rt(
    R: torch.Tensor, t: torch.Tensor, check_rotation: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Invert camera-to-world poses (R, t) into the world-to-camera poses that Colmap stores.

    ``R_w2c, t_w2c = camtoworld_to_worldtocam_Rt(R, t)`` takes camera-to-world rotations ``(B, 3, 3)`` and camera
    centres ``(B, 3, 1)`` and returns the rigid inverse ``(R^T, -R^T @ t)``: the world-to-camera pose of
    `Colmap's images.txt <https://colmap.github.io/format.html#output-format>`_. It transposes rather than inverts, so
    ``R`` must be a rotation; pass ``check_rotation=True`` to verify it. The same formula undoes it, under the name
    :func:`worldtocam_to_camtoworld_Rt`; for a ``(B, 4, 4)`` matrix, use
    :func:`~kornia.geometry.linalg.inverse_transformation`.

    Convention:
        See :doc:`camera and world conventions </get-started/camera-conventions>` for camera-to-world versus
        world-to-camera extrinsics and camera-centre semantics.

        - the returned pair is exactly ``(R^T, -R^T @ t)`` — the **rigid**
          inverse, computed by transposition and never by a matrix inverse, so
          for a proper rotation it inverts the pose to the working dtype's
          rounding
        - what ``t`` means on each side: the input is a camera-to-world pose, so
          its ``t`` is the **camera centre in world coordinates** (the 4x4 sends
          the origin to ``t``); the returned ``-R^T t`` is the **world-to-camera
          translation** of Colmap's ``images.txt`` (the returned 4x4 sends the
          camera centre to the origin)
        - :func:`~kornia.geometry.conversions.worldtocam_to_camtoworld_Rt` is
          the **same function** under the other name: applying either one twice
          returns ``R`` exactly and ``t`` to rounding
        - the shapes are ``(B, 3, 3)`` and ``(B, 3, 1)``, and ``R`` and ``t``
          must carry the same batch size: a mismatched pair raises ``ShapeError``
        - an ``R`` and ``t`` of one integer dtype (``uint8``, ``int8``, ``int16``,
          ``int32`` or ``int64``) are accepted on every device, CUDA (which has
          no integer matmul) included, and the result keeps that dtype, wrapping
          on overflow; ``check_rotation=True`` checks such an ``R`` in ``float32``

    .. warning::
        ``R`` is **assumed** to be a rotation and by default this is not
        checked, so for a non-orthogonal ``R`` the result is a transpose and
        not an inverse, silently: ``R = [[1, 0.5, 0], [0, 1, 0], [0, 0, 2]]``
        gives ``max|M_inv @ M - I| = 3.0``. Pass ``check_rotation=True`` to
        raise a ``ValueError`` instead, which also rejects reflections
        (``det(R) < 0``).

    Args:
        R: Rotation matrix, ``(B, 3, 3)``.
        t: Translation matrix ``(B, 3, 1)``.
        check_rotation: if ``True``, raise ``ValueError`` unless every ``R``
            is a rotation: ``max|R @ R^T - I|`` within 100 eps of the dtype of
            ``R`` (16 eps for float16 and bfloat16) and ``det(R) > 0``.
            Defaults to ``False`` (unchecked).

    Returns:
        Rinv: Rotation matrix, ``(B, 3, 3)``.
        tinv: Translation matrix ``(B, 3, 1)``.

    See Also:
        - :func:`worldtocam_to_camtoworld_Rt`: the reverse direction, the same formula.
        - :func:`~kornia.geometry.linalg.inverse_transformation`: the same rigid inverse of a ``(B, 4, 4)`` matrix.
        - :func:`camtoworld_graphics_to_vision_Rt`: changes the camera axes of a camera-to-world pose instead.
        - :func:`ARKitQTVecs_to_ColmapQTVecs`: an ARKit quaternion pose to Colmap's, through this inversion.

    Example:
        >>> R, t = torch.eye(3)[None], torch.ones(3).reshape(1, 3, 1)
        >>> camtoworld_to_worldtocam_Rt(R, t)
        (tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]]), tensor([[[-1.],
                 [-1.],
                 [-1.]]]))

    """
    KORNIA_CHECK_SHAPE(R, ["B", "3", "3"])
    KORNIA_CHECK_SHAPE(t, ["B", "3", "1"])
    _check_Rt_same_batch(R, t, "camtoworld_to_worldtocam_Rt")

    if check_rotation:
        _check_is_rotation(R, "camtoworld_to_worldtocam_Rt")

    R_inv = R.transpose(1, 2)
    new_t: torch.Tensor = _integer_safe_matmul(-R_inv, t)

    return (R_inv, new_t)


def worldtocam_to_camtoworld_Rt(
    R: torch.Tensor, t: torch.Tensor, check_rotation: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Invert world-to-camera poses (R, t), as Colmap stores them, into camera-to-world poses.

    ``R_c2w, t_c2w = worldtocam_to_camtoworld_Rt(R, t)`` takes world-to-camera rotations ``(B, 3, 3)`` and
    translations ``(B, 3, 1)``, as in Colmap's ``images.txt``, and returns the rigid inverse ``(R^T, -R^T @ t)``: the
    camera-to-world pose, whose translation is the camera centre in world coordinates. It is the same formula as
    :func:`camtoworld_to_worldtocam_Rt`, which documents it; ``R`` must be a rotation, verified with
    ``check_rotation=True``.

    Convention:
        See :doc:`camera and world conventions </get-started/camera-conventions>` for camera-to-world versus
        world-to-camera extrinsics and camera-centre semantics.

        - the **same function** as
          :func:`~kornia.geometry.conversions.camtoworld_to_worldtocam_Rt`,
          which carries the canonical block: for a proper rotation
          ``(R^T, -R^T @ t)`` is its own inverse, so one name serves both
          directions and the choice is documentation for the reader
        - read the other way round here: the input ``t`` is the world-to-camera
          translation and the returned ``-R^T t`` is the camera centre in world
          coordinates
        - the shapes and the rotation warning are as documented there

    Args:
        R: Rotation matrix, ``(B, 3, 3)``.
        t: Translation matrix ``(B, 3, 1)``.
        check_rotation: if ``True``, raise ``ValueError`` unless every ``R``
            is a rotation: ``max|R @ R^T - I|`` within 100 eps of the dtype of
            ``R`` (16 eps for float16 and bfloat16) and ``det(R) > 0``.
            Defaults to ``False`` (unchecked).

    Returns:
        Rinv: Rotation matrix, ``(B, 3, 3)``.
        tinv: Translation matrix ``(B, 3, 1)``.

    See Also:
        - :func:`camtoworld_to_worldtocam_Rt`: the reverse direction, the same formula.
        - :func:`~kornia.geometry.linalg.inverse_transformation`: the same rigid inverse of a ``(B, 4, 4)`` matrix.
        - :func:`ColmapQTVecs_to_ARKitQTVecs`: a Colmap quaternion pose to ARKit's, through this inversion.

    Example:
        >>> R, t = torch.eye(3)[None], torch.ones(3).reshape(1, 3, 1)
        >>> worldtocam_to_camtoworld_Rt(R, t)
        (tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]]), tensor([[[-1.],
                 [-1.],
                 [-1.]]]))

    """
    KORNIA_CHECK_SHAPE(R, ["B", "3", "3"])
    KORNIA_CHECK_SHAPE(t, ["B", "3", "1"])
    _check_Rt_same_batch(R, t, "worldtocam_to_camtoworld_Rt")

    if check_rotation:
        _check_is_rotation(R, "worldtocam_to_camtoworld_Rt")

    R_inv = R.transpose(1, 2)
    new_t: torch.Tensor = _integer_safe_matmul(-R_inv, t)

    return (R_inv, new_t)


def ARKitQTVecs_to_ColmapQTVecs(qvec: torch.Tensor, tvec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert an Apple ARKit camera-to-world screen pose to the world-to-camera pose expected by Colmap.

    ``q_colmap, t_colmap = ARKitQTVecs_to_ColmapQTVecs(q, t)`` takes ARKit camera-to-world poses in the graphics camera
    frame, as ``(w, x, y, z)`` quaternions ``(B, 4)`` and camera centres ``(B, 3, 1)``, and returns the Colmap
    world-to-camera poses in the vision camera frame, the ``QW QX QY QZ TX TY TZ`` of ``images.txt``, as a
    ``(B, 4)`` quaternion and a ``(B, 3, 1)`` translation. ARKit stores quaternions as ``(x, y, z, w)``: reorder them
    first. The sign of the returned quaternion is not canonical. For the reverse, use
    :func:`ColmapQTVecs_to_ARKitQTVecs`.

    Convention:
        See :doc:`camera and world conventions </get-started/camera-conventions>` for the camera frames and
        extrinsics conventions composed by this conversion.

        - **input**: ``qvec`` ``(B, 4)`` in ``(w, x, y, z)`` order —
          ``[1., 0., 0., 0.]`` is the identity — and ``tvec`` ``(B, 3, 1)``,
          read as a **camera-to-world** pose in the **graphics** frame (``+y``
          up, ``-z`` forward), so ``tvec`` is the camera centre in world
          coordinates
        - **output**: ``(q, t)`` in the same layouts, a **world-to-camera** pose
          in the **vision** frame (``+y`` down, ``+z`` forward) — Colmap's
          ``images.txt`` ``QW QX QY QZ TX TY TZ``. The pipeline is
          :func:`~kornia.geometry.conversions.quaternion_to_rotation_matrix`,
          :func:`~kornia.geometry.conversions.camtoworld_graphics_to_vision_Rt`,
          :func:`~kornia.geometry.conversions.camtoworld_to_worldtocam_Rt`, then
          :func:`~kornia.geometry.conversions.rotation_matrix_to_quaternion`
        - ARKit's ``simd_quatf`` exposes ``.vector`` as ``(ix, iy, iz, r)``, i.e.
          ``xyzw``: reorder it to ``wxyz`` first, since this function accepts
          either order silently
        - the input quaternion need not be unit and ``q`` and ``-q`` give the
          same output, as long as ``||q||`` stays above the ``eps = 1e-12``
          normalisation floor and its squared norm does not overflow (about
          ``1.8e19`` in ``float32``); outside that range the result is the
          zero-quaternion answer below or a wrong pose
        - the **sign of the output quaternion is not canonical**: it is the
          representative
          :func:`~kornia.geometry.conversions.rotation_matrix_to_quaternion`'s
          branch produces, so ``[0.5, 0.5, 0.5, 0.5]`` with ``t = (1, 2, 3)``
          returns ``q = [-0.5, -0.5, -0.5, 0.5]`` and ``t = (-2, 3, 1)``.
          Compare rotations, never raw components
        - worked literal: ``q = [0., 1., 0., 1.]``, ``t = (1, 1, 1)`` gives
          ``q = [0.7071, 0., 0.7071, 0.]`` and ``t = (-1, -1, 1)``, the example
          below
        - :func:`~kornia.geometry.conversions.ColmapQTVecs_to_ARKitQTVecs`
          computes the reverse conversion
        - the zero quaternion is not rejected: it rotates as the identity, so
          ``torch.zeros(1, 4)`` with ``t = (1, 1, 1)`` returns
          ``q = [0., 1., 0., 0.]``, ``t = (-1, 1, 1)``, the same answer as the
          identity input
        - ``qvec`` must be real floating point: an integer, bool or complex
          ``qvec`` raises ``TypeCheckError`` (also a ``TypeError``) naming it,
          so cast it first. An integer or bool ``tvec`` promotes to ``qvec``'s
          dtype; a complex ``tvec`` raises ``TypeCheckError``

    Args:
        qvec: ARKit rotation quaternion ``(B, 4)``, [w, x, y, z] format.
        tvec: translation vector ``(B, 3, 1)``, [x, y, z]

    Returns:
        qvec: Colmap rotation quaternion ``(B, 4)``, [w, x, y, z] format.
        tvec: translation vector ``(B, 3, 1)``, [x, y, z]

    See Also:
        - :func:`ColmapQTVecs_to_ARKitQTVecs`: the reverse conversion, Colmap to ARKit.
        - :func:`camtoworld_graphics_to_vision_Rt`: the camera-axes step of this pipeline, for a matrix pose.
        - :func:`camtoworld_to_worldtocam_Rt`: the inversion step of this pipeline, for a matrix pose.

    Example:
        >>> q, t = torch.tensor([0, 1, 0, 1.])[None], torch.ones(3).reshape(1, 3, 1)
        >>> ARKitQTVecs_to_ColmapQTVecs(q, t)
        (tensor([[0.7071, 0.0000, 0.7071, 0.0000]]), tensor([[[-1.0000],
                 [-1.0000],
                 [ 1.0000]]]))

    """
    # ToDo:  integrate QuaterniaonAPI

    if isinstance(qvec, torch.Tensor) and not qvec.is_floating_point():  # complex is not floating point
        raise TypeCheckError(
            f"Input qvec must be a real floating point tensor. Got dtype {qvec.dtype}.", actual_type=qvec.dtype
        )
    # An integer tvec promotes against the floating rotation and gives the float result; only complex is rejected.
    if isinstance(tvec, torch.Tensor) and tvec.is_complex():
        raise TypeCheckError(f"Input tvec must be a real tensor. Got dtype {tvec.dtype}.", actual_type=tvec.dtype)
    Rcg = quaternion_to_rotation_matrix(qvec)
    Rcv, Tcv = camtoworld_graphics_to_vision_Rt(Rcg, tvec)
    R_colmap, t_colmap = camtoworld_to_worldtocam_Rt(Rcv, Tcv)
    t_colmap = t_colmap.reshape(-1, 3, 1)
    q_colmap = rotation_matrix_to_quaternion(R_colmap.contiguous())
    return q_colmap, t_colmap


def ColmapQTVecs_to_ARKitQTVecs(qvec: torch.Tensor, tvec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert a Colmap world-to-camera pose to an Apple ARKit camera-to-world screen pose.

    ``q_arkit, t_arkit = ColmapQTVecs_to_ARKitQTVecs(q, t)`` takes Colmap world-to-camera poses in the vision camera
    frame, as ``(w, x, y, z)`` quaternions ``(B, 4)`` and translations ``(B, 3, 1)``, and returns the ARKit
    camera-to-world poses in the graphics camera frame, as a ``(w, x, y, z)`` quaternion ``(B, 4)`` and the camera
    centre ``(B, 3, 1)``. Reorder the quaternion to ARKit's ``(x, y, z, w)`` before handing it over. The sign of the
    returned quaternion is not canonical. For the reverse, use :func:`ARKitQTVecs_to_ColmapQTVecs`.

    Convention:
        See :doc:`camera and world conventions </get-started/camera-conventions>` for the camera frames and
        extrinsics conventions composed by this conversion.

        - the reverse of
          :func:`~kornia.geometry.conversions.ARKitQTVecs_to_ColmapQTVecs`,
          with the same layouts and the frames exchanged
        - **input**: ``qvec`` ``(B, 4)`` in ``(w, x, y, z)`` order —
          ``[1., 0., 0., 0.]`` is the identity — and ``tvec`` ``(B, 3, 1)``,
          read as a **world-to-camera** pose in the **vision** frame (``+y``
          down, ``+z`` forward) — Colmap's ``images.txt``
          ``QW QX QY QZ TX TY TZ``
        - **output**: ``(q, t)`` in the same layouts, a **camera-to-world**
          pose in the **graphics** frame (``+y`` up, ``-z`` forward), so ``t``
          is the camera centre in world coordinates. The pipeline is
          :func:`~kornia.geometry.conversions.quaternion_to_rotation_matrix`,
          :func:`~kornia.geometry.conversions.worldtocam_to_camtoworld_Rt`,
          :func:`~kornia.geometry.conversions.camtoworld_vision_to_graphics_Rt`,
          then :func:`~kornia.geometry.conversions.rotation_matrix_to_quaternion`
        - the output is ``(w, x, y, z)``; ARKit's ``simd_quatf`` stores
          ``(ix, iy, iz, r)``, i.e. ``xyzw``, so reorder it before handing it
          to ARKit
        - the input quaternion need not be unit and ``q`` and ``-q`` give the
          same output, within the same normalisation range as
          :func:`~kornia.geometry.conversions.ARKitQTVecs_to_ColmapQTVecs`
        - only real floating point ``qvec`` and ``tvec`` are accepted: an
          integer, bool or complex one raises ``TypeCheckError`` (also a
          ``TypeError``) naming the argument, so cast it first
        - the **sign of the output quaternion is not canonical**: a round trip
          through both functions returns the input rotation, but not
          necessarily the input quaternion's sign. Compare rotations, never raw
          components
        - worked literal: the reverse of the
          :func:`~kornia.geometry.conversions.ARKitQTVecs_to_ColmapQTVecs`
          example, ``q = [0.7071, 0., 0.7071, 0.]``, ``t = (-1, -1, 1)``, gives
          ``q = [0., 0.7071, 0., 0.7071]`` and ``t = (1, 1, 1)``, the unit form
          of that example's input; see the example below
        - the zero quaternion is not rejected: it rotates as the identity, so
          ``torch.zeros(1, 4)`` with ``t = (1, 1, 1)`` returns
          ``q = [0., 1., 0., 0.]``, ``t = (-1, -1, -1)``, the same answer as the
          identity input

    Args:
        qvec: Colmap rotation quaternion ``(B, 4)``, [w, x, y, z] format.
        tvec: translation vector ``(B, 3, 1)``, [x, y, z]

    Returns:
        qvec: ARKit rotation quaternion ``(B, 4)``, [w, x, y, z] format.
        tvec: translation vector ``(B, 3, 1)``, [x, y, z]

    See Also:
        - :func:`ARKitQTVecs_to_ColmapQTVecs`: the reverse conversion, ARKit to Colmap.
        - :func:`worldtocam_to_camtoworld_Rt`: the inversion step of this pipeline, for a matrix pose.
        - :func:`camtoworld_vision_to_graphics_Rt`: the camera-axes step of this pipeline, for a matrix pose.

    Example:
        >>> q, t = torch.tensor([0, 1, 0, 1.])[None], torch.ones(3).reshape(1, 3, 1)
        >>> ColmapQTVecs_to_ARKitQTVecs(*ARKitQTVecs_to_ColmapQTVecs(q, t))
        (tensor([[0.0000, 0.7071, 0.0000, 0.7071]]), tensor([[[1.0000],
                 [1.0000],
                 [1.0000]]]))

    """
    for name, value in (("qvec", qvec), ("tvec", tvec)):
        if isinstance(value, torch.Tensor) and not value.is_floating_point():  # complex is not floating point
            raise TypeCheckError(
                f"Input {name} must be a real floating point tensor. Got dtype {value.dtype}.", actual_type=value.dtype
            )
    R_colmap = quaternion_to_rotation_matrix(qvec)
    Rcv, Tcv = worldtocam_to_camtoworld_Rt(R_colmap, tvec)
    Rcg, Tcg = camtoworld_vision_to_graphics_Rt(Rcv, Tcv)
    t_arkit = Tcg.reshape(-1, 3, 1)
    q_arkit = rotation_matrix_to_quaternion(Rcg.contiguous())
    return q_arkit, t_arkit


def vector_to_skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:
    r"""Build the skew-symmetric cross-product matrix ``(3, 3)`` or ``(B, 3, 3)`` of a 3-vector.

    ``S = vector_to_skew_symmetric_matrix(v)`` takes a vector ``(3,)`` or a batch ``(B, 3)`` and returns the matrix
    ``(3, 3)`` or ``(B, 3, 3)`` with ``S @ x == cross(v, x)``. For any number of leading dimensions ``(*, 3)``, use
    :func:`~kornia.geometry.epipolar.cross_product_matrix`, which builds the same matrix.

    A vector :math:`(v1, v2, v3)` has a corresponding skew-symmetric matrix, which is of the form:

    .. math::
        \begin{bmatrix} 0 & -v3 & v2 \\
        v3 & 0 & -v1 \\
        -v2 & v1 & 0\end{bmatrix}

    Convention:
        - the matrix acts on the left as the first cross-product factor:
          ``vector_to_skew_symmetric_matrix(v) @ x`` equals ``cross(v, x)``, not
          its negation — for ``v = (1., 2., 3.)`` and ``x = (4., 5., 6.)`` both
          are ``(-3., 6., -3.)``
        - accepts ``(3,)`` or ``(B, 3)`` only; any higher rank raises
          ``ValueError``

    Args:
        vec: tensor of shape ``(3,)`` or ``(B, 3)``.

    Returns:
        tensor of shape ``(3, 3)`` or ``(B, 3, 3)`` respectively.

    See Also:
        - :func:`~kornia.geometry.epipolar.cross_product_matrix`: the same matrix for ``(*, 3)`` vectors of any rank.
        - :meth:`~kornia.geometry.liegroup.So3.hat`: the same matrix, as the Lie-algebra element of a rotation
          vector.

    Example:
        >>> vec = torch.tensor([1.0, 2.0, 3.0])
        >>> vector_to_skew_symmetric_matrix(vec)
        tensor([[ 0., -3.,  2.],
                [ 3.,  0., -1.],
                [-2.,  1.,  0.]])

    """
    # KORNIA_CHECK_SHAPE(vec, ["B", "3"])
    if vec.shape[-1] != 3 or len(vec.shape) > 2:
        raise ValueError(f"Input vector must be of shape (B, 3) or (3,). Got {vec.shape}")
    v1, v2, v3 = vec[..., 0], vec[..., 1], vec[..., 2]
    zeros = torch.zeros_like(v1)
    return torch.stack(
        [
            torch.stack([zeros, -v3, v2], dim=-1),
            torch.stack([v3, zeros, -v1], dim=-1),
            torch.stack([-v2, v1, zeros], dim=-1),
        ],
        dim=-2,
    )
