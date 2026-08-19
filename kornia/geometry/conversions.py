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

from typing import Optional

import torch
import torch.nn.functional as F

from kornia.constants import pi
from kornia.core._compat import deprecated
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.core.utils import _inverse_3x3_closed_form, _torch_inverse_cast

__all__ = [
    "ARKitQTVecs_to_ColmapQTVecs",
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
    r"""Convert angles from radians to degrees.

    Convention:
        - the input is in **radians** and the output in **degrees**; the
          conversion is elementwise and preserves shape, device and float dtype

    .. warning::
        Two distinct defects, tracked in
        `#3937 <https://github.com/kornia/kornia/issues/3937>`_. A ``float64``
        input is left with only about seven correct significant digits
        (``rad2deg(torch.tensor(math.pi, dtype=torch.float64)) - 180`` is
        ``-5.0e-06``, not ``0``) because ``kornia.constants.pi`` is a
        **float32** tensor — a defect in the constant itself, which several
        other kornia modules also consume (the issue tracks the current
        inventory). Separately, ``rad2deg``/``deg2rad``
        themselves cast that constant to the input dtype, so an integer input
        truncates ``pi`` to ``3``: ``rad2deg(torch.tensor([1, 2, 3]))`` returns
        ``[60., 120., 180.]`` instead of ``[57.2958, 114.5916, 171.8873]``, and
        a ``float64`` constant alone would not fix it.

    Args:
        tensor: torch.Tensor of arbitrary shape.

    Returns:
        torch.Tensor with same shape as input.

    Example:
        >>> input = torch.tensor(3.1415926535)
        >>> rad2deg(input)
        tensor(180.)

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    return 180.0 * tensor / pi.to(tensor.device).type(tensor.dtype)


def deg2rad(tensor: torch.Tensor) -> torch.Tensor:
    r"""Convert angles from degrees to radians.

    Convention:
        - the input is in **degrees** and the output in **radians**; it
          performs the opposite conversion to
          :func:`~kornia.geometry.conversions.rad2deg`

    .. warning::
        Inherits both defects of :func:`~kornia.geometry.conversions.rad2deg`
        — the float32 ``kornia.constants.pi`` and the cast to the input dtype
        (``deg2rad(torch.tensor([180, 90]))`` returns ``[3.0000, 1.5000]``).
        See its warning and
        `#3937 <https://github.com/kornia/kornia/issues/3937>`_.

    Args:
        tensor: torch.Tensor of arbitrary shape.

    Returns:
        tensor with same shape as input.

    Examples:
        >>> input = torch.tensor(180.)
        >>> deg2rad(input)
        tensor(3.1416)

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.0


def pol2cart(rho: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert polar coordinates to cartesian coordinates.

    Convention:
        - the arguments are ``(rho, phi)`` and the return is the tuple
          ``(x, y)``, with ``x = rho * cos(phi)`` and ``y = rho * sin(phi)``:
          ``rho = 5``, ``phi = atan2(4, 3)`` gives ``(3., 4.)``
        - ``phi`` is in **radians** (``rho = 2``, ``phi = pi / 6`` gives
          ``(1.7321, 1.)``) and follows the angle convention documented on
          :func:`~kornia.geometry.conversions.cart2pol`

    Args:
        rho: torch.Tensor of arbitrary shape.
        phi: torch.Tensor of same arbitrary shape.

    Returns:
        - x: torch.Tensor with same shape as input.
        - y: torch.Tensor with same shape as input.

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
    r"""Convert cartesian coordinates to polar coordinates.

    Convention:
        - the arguments are ``(x, y)`` and the return is the tuple
          ``(rho, phi)``: ``(3., 4.)`` gives ``(5., 0.9273)``
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
        - ``rho`` is ``sqrt(x ** 2 + y ** 2 + eps)``, not
          ``sqrt(x ** 2 + y ** 2)`` — see the warning below

    .. warning::
        ``eps`` is added *inside* the square root, so the expression evaluated
        is ``sqrt(x ** 2 + y ** 2 + eps)`` and ``rho`` is biased high. Whether
        that bias survives the rounding of the working dtype depends on where
        it is measured. Away from the origin it is usually invisible:
        ``cart2pol(3., 4.)`` returns ``5.000000001`` in ``float64`` but rounds
        back to exactly ``5.`` in ``float32`` and ``float16``. At the origin it
        is the whole answer: ``cart2pol(torch.tensor(0.), torch.tensor(0.))``
        returns ``rho = 9.9999997e-05`` in ``float32`` and ``1e-04`` in
        ``float64`` rather than ``0`` (in ``float16``, ``eps`` underflows the
        sum and ``rho`` is ``0.``). Tracked in
        `#3939 <https://github.com/kornia/kornia/issues/3939>`_.

    Args:
        x: torch.Tensor of arbitrary shape.
        y: torch.Tensor of same arbitrary shape.
        eps: added inside the square root when computing ``rho``. A positive
            ``eps`` that is representable in the working dtype keeps the
            gradient of ``rho`` finite at the origin, where it is ``nan`` with
            ``eps=0``. The default ``1e-8`` underflows in ``float16`` (see the
            warning above), so there the origin gradient is still ``nan``.

    Returns:
        - rho: torch.Tensor with same shape as input.
        - phi: torch.Tensor with same shape as input.

    Example:
        >>> x = torch.rand(1, 3, 3)
        >>> y = torch.rand(1, 3, 3)
        >>> rho, phi = cart2pol(x, y)

    """
    if not (isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor)):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(x)}, {type(y)}")

    rho = torch.sqrt(x**2 + y**2 + eps)
    phi = torch.atan2(y, x)
    return rho, phi


def convert_points_from_homogeneous(points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert points from homogeneous to Euclidean space.

    Convention:
        - the **last** component of the last dimension is the homogeneous
          coordinate ``w`` and is dropped from the output
        - requires rank :math:`\geq 2`: a bare :math:`(D,)` point raises
          ``ValueError``
        - when ``abs(w) > eps`` the remaining components are divided by ``w``
          with its sign kept (``[[2., 4., -2.]]`` gives ``[[-1., -2.]]``), up to
          the bias described in the warning below
        - when ``abs(w) <= eps`` (default ``eps = 1e-8``; the test is a strict
          ``>``) the numerator is instead returned **unchanged**, following
          OpenCV: ``[[2., 4., 0.]]`` gives ``[[2., 4.]]``

    .. warning::
        The division is by ``w + eps`` rather than by ``w``, and ``eps`` is
        added without regard to the sign of ``w``, so the **signed** relative
        error of the result is exactly ``-eps / (w + eps)``. At ``w = 2e-8``
        that is ``-1/3``: the exact result ``[1e8, 2e8]`` comes out as
        ``[6.67e7, 1.33e8]`` (33 % low). At ``w = -2e-8`` it is ``+1``:
        ``[-1e8, -2e8]`` comes out as ``[-2e8, -4e8]`` (100 % high). Only for
        ``abs(w)`` much larger than ``eps`` does it reduce to the familiar
        ``-eps / w``, and there it is usually below the rounding of the working
        dtype — at ``w = 2`` the measured error is ``-5.0e-09`` in ``float64``,
        while in ``float32`` ``2 + eps`` rounds back to ``2`` and the result is
        exact. The numbers above assume ``eps`` is representable: in
        ``float16`` both the default ``eps`` and ``w = 2e-8`` underflow to
        ``0``, so ``[[2., 4., 2e-8]]`` takes the ``abs(w) <= eps`` pass-through
        branch and returns ``[[2., 4.]]``. Tracked in
        `#3938 <https://github.com/kornia/kornia/issues/3938>`_.

    Args:
        points: the points to be transformed of shape :math:`(*, N, D)`.
        eps: to avoid division by zero.

    Returns:
        the points in Euclidean space :math:`(*, N, D-1)`.

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
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

    return scale * points[..., :-1]


def convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    r"""Convert points from Euclidean to homogeneous space.

    Convention:
        - appends the constant ``1`` as the **last** component of the last
          dimension, which is where
          :func:`~kornia.geometry.conversions.convert_points_from_homogeneous`
          expects it
        - requires rank :math:`\geq 2`: a bare :math:`(D,)` point raises
          ``ValueError``
        - the input dtype is preserved, integer dtypes included

    Args:
        points: the points to be transformed with shape :math:`(*, N, D)`.

    Returns:
        the points in homogeneous coordinates :math:`(*, N, D+1)`.

    Examples:
        >>> input = torch.tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        tensor([[0., 0., 1.]])

    """
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(points)}")
    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    return F.pad(points, [0, 1], "constant", 1.0)


def _convert_affinematrix_to_homography_impl(A: torch.Tensor) -> torch.Tensor:
    H: torch.Tensor = F.pad(A, [0, 0, 0, 1], "constant", value=0.0)
    H[..., -1, -1] += 1.0
    return H


def convert_affinematrix_to_homography(A: torch.Tensor) -> torch.Tensor:
    r"""Convert batch of affine matrices.

    Convention:
        - appends the row ``[0, 0, 1]`` at the bottom and copies the
          :math:`2 \times 3` block verbatim; the input tensor is not modified
        - the rank is enforced to be exactly 3, so batching is **mandatory**: an
          unbatched :math:`(2, 3)` matrix raises ``ValueError``

    Args:
        A: the affine matrix with shape :math:`(B,2,3)`.

    Returns:
         the homography matrix with shape of :math:`(B,3,3)`.

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
    r"""Convert batch of 3d affine matrices.

    Convention:
        - same as
          :func:`~kornia.geometry.conversions.convert_affinematrix_to_homography`,
          except that ``A`` is :math:`(B, 3, 4)` and the appended bottom row is
          ``[0, 0, 0, 1]``

    Args:
        A: the affine matrix with shape :math:`(B,3,4)`.

    Returns:
         the homography matrix with shape of :math:`(B,4,4)`.

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
    r"""Convert 3d vector of axis-angle rotation to 3x3 rotation matrix.

    Convention:
        - the input is the rotation axis scaled by the angle, in **radians**,
          and must be batched — a bare ``(3,)`` vector raises (see the shape
          warning below): ``[[0., 0., pi/2]]`` is a quarter turn about ``+z``,
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
          to this function, returning an equal result — see the alias warning
          below

    .. warning::
        The returned matrix is **not orthogonal**: ``eps = 1e-6`` is added to
        the angle when the axis is normalised, which shrinks the axis. In
        ``float64`` at ``theta = pi/2`` about ``+z``, ``det(R)`` is
        ``0.9999974535249636`` and ``max|R R^T - I|`` is
        ``2.5464750363912714e-06``; the determinant is axis-independent to the
        last digit or two, while the orthogonality residual is not (a generic
        axis gives
        ``2.091747640764474e-06``). ``float32`` is no better
        (``2.5033950805664062e-06`` on the same input), and
        the second example below hides it — the printed ``1.0000e+00`` at
        ``R[0, 0]`` is really ``0.9999987483024597``. Below an internal
        threshold on ``theta ** 2`` the first-order matrix
        ``[[1, -rz, ry], [rz, 1, -rx], [-ry, rx, 1]]`` is returned instead, with
        ``det = 1 + theta ** 2``: in ``float64`` the input ``[0., 0., 1e-3]``
        takes that branch (``det = 1.000001``) while ``1e-3 * (1, 2, 3)/sqrt(14)``
        does not (``det = 0.9999999970044947``), so which branch an input takes
        depends on its axis and dtype. Tracked in
        `#3947 <https://github.com/kornia/kornia/issues/3947>`_.

    .. warning::
        Only rank-2 input is accepted, despite the guard's ``(*, 3)`` message:
        ``(3,)`` raises ``IndexError: Dimension out of range``, ``(2, 5, 3)``
        raises ``ValueError: too many values to unpack (expected 3)`` and
        ``(1, 1, 3)`` raises ``ValueError: not enough values to unpack``.
        Composing with
        :func:`~kornia.geometry.conversions.rotation_matrix_to_axis_angle`
        therefore fails for every rotation-matrix rank but 3. Tracked in
        `#3955 <https://github.com/kornia/kornia/issues/3955>`_.

    .. warning::
        Calling any of this module's four deprecated aliases
        (``angle_axis_to_rotation_matrix``, ``rotation_matrix_to_angle_axis``,
        ``quaternion_to_angle_axis``, ``angle_axis_to_quaternion``) rewrites the
        process-global ``DeprecationWarning`` filters, so
        ``-W error::DeprecationWarning`` does not turn the warning into an error
        and every later ``DeprecationWarning`` raised in the process is affected
        too. Tracked in
        `#3956 <https://github.com/kornia/kornia/issues/3956>`_.

    Args:
        axis_angle: tensor of 3d vector of axis-angle rotations in radians with shape :math:`(N, 3)`.

    Returns:
        tensor of rotation matrices of shape :math:`(N, 3, 3)`.

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

    def _compute_rotation_matrix(axis_angle: torch.Tensor, theta2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        theta = torch.sqrt(theta2.clamp(min=1e-12))  # clamping to ensure no nan gradients
        wxyz = axis_angle / (theta.unsqueeze(-1) + eps)  # (B, 3)
        wx, wy, wz = wxyz.unbind(dim=1)  # (B,)

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

        rot = torch.stack(
            [
                torch.stack([r00, r01, r02], dim=-1),
                torch.stack([r10, r11, r12], dim=-1),
                torch.stack([r20, r21, r22], dim=-1),
            ],
            dim=1,
        )

        return rot

    def _compute_rotation_matrix_taylor(axis_angle: torch.Tensor) -> torch.Tensor:
        rx, ry, rz = axis_angle.unbind(-1)
        k_one = torch.ones_like(rx)

        rot = torch.stack(
            [
                k_one,
                -rz,
                ry,
                rz,
                k_one,
                -rx,
                -ry,
                rx,
                k_one,
            ],
            dim=-1,
        ).view(-1, 3, 3)

        return rot

    theta2 = (axis_angle * axis_angle).sum(dim=-1)

    rot_normal = _compute_rotation_matrix(axis_angle, theta2)  # (N,3,3)
    rot_taylor = _compute_rotation_matrix_taylor(axis_angle)  # (N,3,3)

    mask = (theta2 > 1e-6).view(-1, 1, 1)  # shape (N,1,1)

    rotation_matrix = torch.where(mask, rot_normal, rot_taylor)

    return rotation_matrix


@deprecated(replace_with="axis_angle_to_rotation_matrix", version="0.7.0")
def angle_axis_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:  # noqa: D103
    return axis_angle_to_rotation_matrix(axis_angle)


def rotation_matrix_to_axis_angle(rotation_matrix: torch.Tensor) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to Rodrigues vector in radians.

    Convention:
        - any number of leading batch dimensions is accepted: :math:`(3, 3)`
          gives :math:`(3,)` and :math:`(2, 5, 3, 3)` gives :math:`(2, 5, 3)`
        - the output is the rotation axis scaled by the angle in **radians**,
          the parametrization
          :func:`~kornia.geometry.conversions.axis_angle_to_rotation_matrix`
          consumes — but that function accepts only rank-2 input, so the
          :math:`(3,)` and :math:`(2, 5, 3)` results above cannot be fed
          straight back (see its shape warning)
        - the round trip through
          :func:`~kornia.geometry.conversions.axis_angle_to_rotation_matrix` is
          accurate only to about ``1e-6`` even in ``float64`` — measured
          ``8.0e-07`` at ``theta = 1e-3`` and ``5.4e-07`` at ``theta = pi``
          about ``(1, 2, 3)/sqrt(14)`` — because of that function's
          `#3947 <https://github.com/kornia/kornia/issues/3947>`_
        - the input is **not** checked for being a rotation matrix:
          ``zeros(3, 3)`` returns ``[0., 0., 3.1416]``, ``2 * eye(3)`` returns
          ``[0., 0., 0.]``, and the reflection ``diag(-1, 1, 1)``
          (``det = -1``) also returns ``[0., 0., 0.]``, i.e. is silently
          reported as no rotation at all
        - ``rotation_matrix_to_angle_axis`` is the deprecated alias of this
          function since 0.7.0; see the alias warning on
          :func:`~kornia.geometry.conversions.axis_angle_to_rotation_matrix`

    Args:
        rotation_matrix: rotation matrix of shape :math:`(*, 3, 3)`.

    Returns:
        Rodrigues vector transformation of shape :math:`(*, 3)`.

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
    quaternion: torch.Tensor = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_axis_angle(quaternion)


@deprecated(replace_with="rotation_matrix_to_axis_angle", version="0.7.0")
def rotation_matrix_to_angle_axis(rotation_matrix: torch.Tensor) -> torch.Tensor:  # noqa: D103
    return rotation_matrix_to_axis_angle(rotation_matrix)


def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in (w, x, y, z) format.

    Convention:
        - the returned components are ``(w, x, y, z)``, real part first — the
          layout, and its silent misreading, are spelled out on
          :func:`~kornia.geometry.conversions.quaternion_to_rotation_matrix`
        - the sign is **not** canonicalised to ``w >= 0``. For
          ``trace(R) > 0`` the returned ``w`` is non-negative, but for
          ``trace(R) <= 0`` it is the dominant of ``x, y, z`` that is forced
          non-negative and ``w`` may be negative: 170 degrees about
          ``(1, 2, -3)/sqrt(14)`` (``trace = -0.9696``) returns
          ``[-0.0872, -0.2662, -0.5325, 0.7987]``
        - the input is **not** checked for being a rotation matrix; see the
          degenerate inputs listed on
          :func:`~kornia.geometry.conversions.rotation_matrix_to_axis_angle`

    .. warning::
        ``eps`` is added *inside* the square root that produces the dominant
        component, so with the default the result is not a unit quaternion. In
        ``float64``, ``rotation_matrix_to_quaternion(torch.eye(3))`` returns
        ``[1.0000000012499999, 0., 0., 0.]`` (``||q|| - 1`` is ``1.25e-09``),
        while ``eps=0.0`` returns exactly ``[1., 0., 0., 0.]``. The inflation is
        below one ulp of 1.0 in ``float32``, ``float16`` and ``bfloat16``, where
        the identity already comes back exactly unit. Tracked in
        `#3951 <https://github.com/kornia/kornia/issues/3951>`_.

    Args:
        rotation_matrix: the rotation matrix to convert with shape :math:`(*, 3, 3)`.
        eps: added inside the square root of the dominant component; see the warning above.

    Return:
        the rotation in quaternion with shape :math:`(*, 4)`.

    Example:
        >>> input = torch.tensor([[1., 0., 0.],
        ...                       [0., 1., 0.],
        ...                       [0., 0., 1.]])
        >>> rotation_matrix_to_quaternion(input, eps=torch.finfo(input.dtype).eps)
        tensor([1., 0., 0., 0.])

    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec: torch.Tensor = rotation_matrix.reshape(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond() -> torch.Tensor:
        sq = torch.sqrt(trace + 1.0 + eps) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1() -> torch.Tensor:
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2() -> torch.Tensor:
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3() -> torch.Tensor:
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0  # sq = 4 * qz.
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
    r"""Normalize a quaternion.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Convention:
        - it is an L2 normalisation of the **last** axis and nothing else, which
          is why the ``(x, y, z, w)`` or ``(w, x, y, z)`` phrasing above is
          accurate here: ``[4., 3., 2., 1.]`` normalises to the reversal of what
          ``[1., 2., 3., 4.]`` gives
        - the sign is preserved — ``normalize_quaternion(-q)`` is exactly
          ``-normalize_quaternion(q)``, there is no ``w >= 0``
          canonicalisation

    .. warning::
        When ``||q|| < eps`` the output is **not** a unit quaternion and no
        error is raised: in ``float64`` with the default ``eps = 1e-12``,
        ``[1e-13, 0., 0., 0.]`` returns ``[0.1, 0., 0., 0.]`` and ``zeros(4)``
        returns ``zeros(4)``, and with ``eps=0.0`` the zero quaternion returns
        ``[nan, nan, nan, nan]`` instead. ``float32`` and ``bfloat16`` behave
        the same up to their rounding (``0.10000000149011612`` and
        ``0.099609375`` for the first input). In ``float16`` both ``1e-13`` and
        the default ``eps`` round to ``0``, so the clamp is a no-op and each of
        those two inputs already returns ``[nan, nan, nan, nan]`` at the
        default — the same underflow class as
        `#3966 <https://github.com/kornia/kornia/issues/3966>`_. Tracked in
        `#3952 <https://github.com/kornia/kornia/issues/3952>`_.

    Args:
        quaternion: a tensor containing a quaternion to be normalized.
          The tensor can be of shape :math:`(*, 4)`.
        eps: floor on the norm used as the divisor; see the warning above.

    Return:
        the normalized quaternion of shape :math:`(*, 4)`.

    Example:
        >>> quaternion = torch.tensor((1., 0., 1., 0.))
        >>> normalize_quaternion(quaternion)
        tensor([0.7071, 0.0000, 0.7071, 0.0000])

    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")
    return F.normalize(quaternion, p=2.0, dim=-1, eps=eps)


# based on:
# https://github.com/matthew-brett/transforms3d/blob/8965c48401d9e8e66b6a8c37c65f2fc200a076fa/transforms3d/quaternions.py#L101
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py#L247


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    r"""Convert a quaternion to a rotation matrix.

    The quaternion should be in (w, x, y, z) format.

    Convention:
        - the quaternion is ``(w, x, y, z)``, **real part first**, shape
          :math:`(*, 4)` in and :math:`(*, 3, 3)` out. ``[1., 0., 0., 0.]``
          gives the identity and ``[0., 0., 0., 1.]`` gives
          ``[[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]``, a half turn about
          ``+z`` — so a caller passing ``(x, y, z, w)`` gets a valid-looking
          rotation matrix and no error
        - ``q`` and ``-q`` are the same rotation (double cover) and return
          bit-identical matrices
        - the input is normalised internally:
          ``quaternion_to_rotation_matrix(2 * q)`` is bitwise
          ``quaternion_to_rotation_matrix(q)`` (400 000 random unit
          quaternions, at every dtype). Rescaling by random factors between
          ``1e-6`` and ``1e6`` moves the matrix only by the working dtype's
          rounding in ``float64`` and ``float32`` (by ``1.3e-15`` and
          ``6.0e-07`` over 2000 random unit quaternions), but the
          reduced-precision dtypes do **not** hold to that: ``bfloat16`` moves
          by ``3.1e-02``, and ``float16`` loses the property altogether — near
          the top of that range the rescaled matrix bears no relation to the
          input at all (entries live in ``[-1, 1]``, and the deviation
          approaches the maximum possible ``2``), while the ``1e6`` end is
          ``nan`` outright because ``0.5 * 1e6`` overflows the dtype.
          Once ``||q||`` drops below
          :func:`~kornia.geometry.conversions.normalize_quaternion`'s
          ``eps = 1e-12`` the clamp takes over and rescaling changes the matrix
          outright: in ``float64``, ``q * 1e-13`` and ``q`` can give matrices
          that differ by order 1
        - :func:`~kornia.geometry.conversions.quaternion_to_axis_angle` is
          scale-safe over its own range, given in its Convention block; the two
          functions that are **not** are
          :func:`~kornia.geometry.conversions.quaternion_exp_to_log` and
          :func:`~kornia.geometry.conversions.euler_from_quaternion`, whose
          warnings give the measured errors
        - applied on the left to a column vector, ``+theta`` about ``+z`` maps
          ``x_hat`` to ``y_hat`` (right-hand rule)
        - the output dtype follows the input at every shape but one: an
          **unbatched** ``float16`` or ``bfloat16`` quaternion of shape
          ``(4,)`` returns a ``float32`` matrix, so the same call on ``q`` and
          on ``q[None]`` returns two different dtypes — see the warning below.
          :func:`~kornia.geometry.conversions.normalize_quaternion`,
          :func:`~kornia.geometry.conversions.quaternion_to_axis_angle` and
          :func:`~kornia.geometry.conversions.quaternion_exp_to_log` return the
          input dtype at every shape

    .. warning::
        An **unbatched** ``float16`` or ``bfloat16`` quaternion of shape
        ``(4,)`` returns a ``float32`` matrix; the same quaternion batched —
        ``(1, 4)``, ``(2, 4)``, ``(3, 3, 4)`` — returns the input dtype. The
        three diagonal entries are computed against a 0-dim ``float32``
        literal, and type promotion ranks a dimensioned tensor above a 0-dim
        one: batched components therefore keep their dtype, while the 0-dim
        components of an unbatched input tie with the literal and ``float32``
        wins the category. Tracked in
        `#3954 <https://github.com/kornia/kornia/issues/3954>`_.

    .. warning::
        The zero quaternion returns the identity matrix rather than raising:
        ``quaternion_to_rotation_matrix(torch.zeros(4))`` is ``eye(3)`` in
        ``float64``, ``float32`` and ``bfloat16``. In ``float16`` the internal
        ``eps = 1e-12`` normalisation floor rounds to ``0``, the guard it
        provides disappears and the matrix is all-``nan`` instead — the same
        underflow class as
        `#3966 <https://github.com/kornia/kornia/issues/3966>`_. Tracked in
        `#3952 <https://github.com/kornia/kornia/issues/3952>`_.

    Args:
        quaternion: a tensor containing a quaternion to be converted.
          The tensor can be of shape :math:`(*, 4)`.

    Return:
        the rotation matrix of shape :math:`(*, 3, 3)`.

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
    one: torch.Tensor = torch.tensor(1.0)

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
    matrix = matrix_flat.reshape(output_shape)

    return matrix


def quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to axis angle of rotation in radians.

    The quaternion should be in (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Convention:
        - the input is ``(w, x, y, z)``, real part first (see
          :func:`~kornia.geometry.conversions.quaternion_to_rotation_matrix`),
          and the output is the rotation axis scaled by the angle in
          **radians**
        - for ``w != 0`` the double cover is collapsed: ``q`` and ``-q`` return
          the same vector, the representative with ``|theta| <= pi``, and in
          ``float64`` the two are bit-identical (400 000 random finite
          quaternions). At lower precision they agree to the working dtype's
          rounding
        - at exactly ``w = 0`` — a half turn, and ``0.`` is exactly
          representable — the collapse does not happen and the two return exact
          negations: ``[0., 1., 0., 0.]`` gives ``[pi, 0., 0.]`` while
          ``[-0., -1., 0., 0.]`` gives ``[-pi, 0., 0.]``, the same rotation
          written the other way round
        - the input need not be unit: ``quaternion_to_axis_angle(2 * q)`` is
          bitwise ``quaternion_to_axis_angle(q)`` (400 000 random unit
          quaternions, at every dtype). Rescaling by random factors between
          ``1e-6`` and ``1e6`` moves the result only by the working dtype's
          rounding in ``float64`` and ``float32`` (by ``8.9e-16`` and
          ``4.8e-07`` over 2000 random unit quaternions), but the
          reduced-precision dtypes do **not** hold to that: ``bfloat16`` moves
          by ``3.1e-02`` and ``float16`` by over ``3`` radians, and the ``1e6``
          end is ``nan`` at ``float16`` because ``0.5 * 1e6`` overflows the
          dtype.
          At extreme scales the squares of the vector part underflow and the
          result does change: in ``float32``,
          ``quaternion_to_axis_angle(q * 1e-25)`` returns exactly
          ``2 * (q * 1e-25)[..., 1:]``
        - ``quaternion_to_angle_axis`` is the deprecated alias of this function
          since 0.7.0; see the alias warning on
          :func:`~kornia.geometry.conversions.axis_angle_to_rotation_matrix`

    .. warning::
        The gradient at the identity quaternion is ``nan``:
        ``quaternion_to_axis_angle(torch.tensor([1., 0., 0., 0.])).sum().backward()``
        leaves ``[nan, nan, nan, nan]`` in ``.grad``, from an unclamped
        ``sqrt(0)``. Away from the identity the gradient is finite. Tracked in
        `#3949 <https://github.com/kornia/kornia/issues/3949>`_.

    Args:
        quaternion: tensor with quaternions.

    Return:
        tensor with axis angle of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

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

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta)
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    axis_angle: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    axis_angle[..., 0] += q1 * k
    axis_angle[..., 1] += q2 * k
    axis_angle[..., 2] += q3 * k
    return axis_angle


@deprecated(replace_with="quaternion_to_axis_angle", version="0.7.0")
def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:  # noqa: D103
    return quaternion_to_axis_angle(quaternion)


def quaternion_log_to_exp(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    r"""Apply exponential map to log quaternion.

    Convention:
        - the **input** is a 3-vector of shape :math:`(*, 3)`, not a
          quaternion; the **output** is the quaternion of shape :math:`(*, 4)`
          in ``(w, x, y, z)`` order (see
          :func:`~kornia.geometry.conversions.quaternion_to_rotation_matrix`)
        - the map is ``w = cos(||v||)`` with vector part
          ``sin(||v||) * v / ||v||``, so whenever ``||v||`` is computed finitely
          the output is a unit quaternion up to the working dtype's rounding.
          Two inputs escape that: the ``float16`` zero vector, and any ``v``
          large enough that ``||v||`` overflows — see the two warnings below
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

    .. warning::
        In ``float16`` the default ``eps = 1e-8`` rounds to ``0``, so the clamp
        that guards the division is a no-op and the zero vector returns
        ``[1., nan, nan, nan]``. It is the only input that does: ``torch.norm``
        does not underflow at ``float16``, so every ``v`` that is not exactly
        zero has a non-zero norm and is unaffected. ``float64``, ``float32``
        and ``bfloat16`` return ``[1., 0., 0., 0.]`` for ``zeros(3)``, and so
        does ``float16`` with a representable ``eps`` (e.g. ``eps=1e-3``).
        Tracked in
        `#3966 <https://github.com/kornia/kornia/issues/3966>`_.

    .. warning::
        ``||v||`` is computed with ``torch.norm(p=2)``, which forms the sum of
        squares and so overflows to ``inf`` far below the largest finite input.
        Beyond that point **all four** components come back ``nan``, even
        though the exponential map of a large finite vector is a perfectly good
        unit quaternion. Where the turnover sits depends on the accumulator:
        ``float32`` and ``bfloat16`` accumulate in their own dtype and so
        overflow once ``||v||`` passes ``sqrt(finfo.max)`` — from about
        ``1.8446744e19`` in ``float32`` (against a ``finfo.max`` of ``3.4e38``)
        and about ``1.841e19`` in ``bfloat16`` — as does ``float64``, from
        about ``1.3407808e154`` (against ``1.8e308``). ``float16`` accumulates
        in wider precision, so its squares never overflow and it turns over
        only once the true ``||v||`` exceeds what ``float16`` itself can hold,
        near ``65520``. That still happens, but it takes **two or more**
        non-zero components, since a single one cannot exceed ``65504``:
        ``[37824., 37824., 37824.]`` is finite while
        ``[37856., 37856., 37856.]`` is all ``nan``. Tracked in
        `#3975 <https://github.com/kornia/kornia/issues/3975>`_.

    Args:
        quaternion: the log quaternion, a tensor of shape :math:`(*, 3)`.
        eps: a small number for clamping.

    Return:
        the quaternion exponential map of shape :math:`(*, 4)`.

    Example:
        >>> quaternion = torch.tensor((0., 0., 0.))
        >>> quaternion_log_to_exp(quaternion, eps=torch.finfo(quaternion.dtype).eps)
        tensor([1., 0., 0., 0.])

    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 3:
        raise ValueError(f"Input must be a tensor of shape (*, 3). Got {quaternion.shape}")

    # compute quaternion norm
    norm_q: torch.Tensor = torch.norm(quaternion, p=2, dim=-1, keepdim=True).clamp(min=eps)

    # compute scalar and vector
    quaternion_vector: torch.Tensor = quaternion * torch.sin(norm_q) / norm_q
    quaternion_scalar: torch.Tensor = torch.cos(norm_q)

    # compose quaternion and return
    quaternion_exp = torch.cat((quaternion_scalar, quaternion_vector), dim=-1)

    return quaternion_exp


def quaternion_exp_to_log(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    r"""Apply the log map to a quaternion.

    The quaternion should be in (w, x, y, z) format.

    Convention:
        - the input is the quaternion of shape :math:`(*, 4)` in
          ``(w, x, y, z)`` order and the output is a 3-vector of shape
          :math:`(*, 3)`, the argument
          :func:`~kornia.geometry.conversions.quaternion_log_to_exp` takes
        - for a **unit** ``q`` on the ``w >= 0`` half of the double cover the
          result is ``quaternion_to_axis_angle(q) / 2``. On the ``w < 0`` half the two
          part company: this function applies ``acos(w)`` as given, while
          :func:`~kornia.geometry.conversions.quaternion_to_axis_angle`
          collapses the double cover. For the ``q`` whose log is ``v``,
          ``quaternion_exp_to_log(-q)`` is ``-(pi - ||v||) * v / ||v||``
        - ``quaternion_exp_to_log(quaternion_log_to_exp(v))`` returns ``v`` for
          ``0 < ||v|| < pi`` and the wrapped ``||v|| - 2 * pi`` above ``pi``
          (``pi + 0.5`` comes back as ``-2.6415926535897936``). At exactly
          ``||v|| = pi``, and only in ``float64``, it collapses to ``3.85e-08``
          — there ``cos(pi)`` rounds to ``-1`` and the vector part falls under
          the ``eps`` clamp. Both are properties of the map, not defects

    .. warning::
        The input is **not** normalised, so a non-unit quaternion is silently
        given a wrong log: ``[0.5, 0.5, 0., 0.]`` returns
        ``[1.0471975511965976, 0., 0.]``, 33 % larger than the
        ``[0.7853981633974484, 0., 0.]`` of the same rotation normalised, and
        ``[2., 0., 0., 0.]`` returns the origin because ``w`` is clamped to
        ``1``. Tracked in
        `#3953 <https://github.com/kornia/kornia/issues/3953>`_.

    .. warning::
        In ``float16`` the default ``eps = 1e-8`` underflows to ``0``, so the
        clamp that guards the division is a no-op and **any** quaternion with a
        zero vector part returns ``[nan, nan, nan]`` — including the identity
        ``[1., 0., 0., 0.]``, whose log is the origin at every other dtype.
        Passing a representable ``eps`` (e.g. ``eps=1e-3``) returns
        ``[0., 0., 0.]`` there. Tracked in
        `#3966 <https://github.com/kornia/kornia/issues/3966>`_.

    Args:
        quaternion: a tensor containing a quaternion to be converted.
          The tensor can be of shape :math:`(*, 4)`.
        eps: a small number for clamping.

    Return:
        the quaternion log map of shape :math:`(*, 3)`.

    Example:
        >>> quaternion = torch.tensor((1., 0., 0., 0.))
        >>> quaternion_exp_to_log(quaternion, eps=torch.finfo(quaternion.dtype).eps)
        tensor([0., 0., 0.])

    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")

    # unpack quaternion vector and scalar
    quaternion_scalar = quaternion[..., 0:1]
    quaternion_vector = quaternion[..., 1:4]

    # compute quaternion norm
    norm_q: torch.Tensor = torch.norm(quaternion_vector, p=2, dim=-1, keepdim=True).clamp(min=eps)

    # apply log map
    quaternion_log: torch.Tensor = (
        quaternion_vector * torch.acos(torch.clamp(quaternion_scalar, min=-1.0, max=1.0)) / norm_q
    )

    return quaternion_log


# based on:
# https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py#L138


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    r"""Convert an axis angle to a quaternion.

    The quaternion vector has components in (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Convention:
        - the input is the rotation axis scaled by the angle in **radians**,
          shape :math:`(*, 3)`; the output is ``(w, x, y, z)`` with
          ``w = cos(theta / 2)``, shape :math:`(*, 4)` (see
          :func:`~kornia.geometry.conversions.quaternion_to_rotation_matrix`
          for the layout)
        - nothing is canonicalised, so ``theta > pi`` returns the ``w < 0`` half
          of the double cover: ``[2 * pi, 0., 0.]`` gives ``[-1., 0., 0., 0.]``
        - the round trip with
          :func:`~kornia.geometry.conversions.quaternion_to_axis_angle` is exact
          in ``float64``: the measured error is ``0`` to ``2.2e-16`` at
          ``theta = 0``, ``1e-3``, ``0.7``, ``2`` and ``pi`` about
          ``(1, 2, 3)/sqrt(14)``
        - ``angle_axis_to_quaternion`` is the deprecated alias of this function
          since 0.7.0; see the alias warning on
          :func:`~kornia.geometry.conversions.axis_angle_to_rotation_matrix`

    .. warning::
        The gradient at the zero rotation is ``nan``:
        ``axis_angle_to_quaternion(torch.zeros(3)).sum().backward()`` leaves
        ``[nan, nan, nan]`` in ``.grad``, from an unclamped ``sqrt(0)``.
        Tracked in `#3949 <https://github.com/kornia/kornia/issues/3949>`_.

    .. warning::
        An integer tensor returns an all-zero integer tensor instead of a
        quaternion: ``axis_angle_to_quaternion(torch.tensor([1, 0, 0]))`` is
        ``tensor([0, 0, 0, 0])`` of dtype ``int64``, against the ``float32``
        answer ``[0.8776, 0.4794, 0., 0.]``, because the output buffer is
        allocated with the input dtype. Tracked in
        `#3948 <https://github.com/kornia/kornia/issues/3948>`_.

    Args:
        axis_angle: tensor with axis angle in radians.

    Return:
        tensor with quaternion.

    Shape:
        - Input: :math:`(*, 3)` where `*` means, any number of dimensions
        - Output: :math:`(*, 4)`

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

    theta: torch.Tensor = torch.sqrt(theta_squared)
    half_theta: torch.Tensor = theta * 0.5

    mask: torch.Tensor = theta_squared > 0.0
    ones: torch.Tensor = torch.ones_like(half_theta)

    k_neg: torch.Tensor = 0.5 * ones
    k_pos: torch.Tensor = torch.sin(half_theta) / theta
    k: torch.Tensor = torch.where(mask, k_pos, k_neg)
    w: torch.Tensor = torch.where(mask, torch.cos(half_theta), ones)

    quaternion: torch.Tensor = torch.zeros(
        size=(*axis_angle.shape[:-1], 4), dtype=axis_angle.dtype, device=axis_angle.device
    )
    quaternion[..., 1:2] = a0 * k
    quaternion[..., 2:3] = a1 * k
    quaternion[..., 3:4] = a2 * k
    quaternion[..., 0:1] = w
    return quaternion


@deprecated(replace_with="axis_angle_to_quaternion", version="0.7.0")
def angle_axis_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:  # noqa: D103
    return axis_angle_to_quaternion(axis_angle)


# inspired by: https://stackoverflow.com/questions/56207448/efficient-quaternions-to-euler-transformation


def euler_from_quaternion(
    w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a quaternion coefficients to Euler angles.

    Convention:
        - the four quaternion coefficients are passed as **separate** tensors in
          ``(w, x, y, z)`` order, not as one :math:`(*, 4)` tensor, and their
          shapes must be exactly equal: broadcastable shapes such as ``(2,)``
          and ``(1,)`` raise ``BaseError: Validation condition failed``, and
          Python floats raise ``AttributeError``
        - the return is a tuple of three tensors ``(roll, pitch, yaw)`` in
          **radians**, with ``roll`` about ``x``, ``pitch`` about ``y`` and
          ``yaw`` about ``z``; the composition they stand for is documented on
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

    .. warning::
        At ``pitch = ±pi/2`` the returned triple usually does not represent the
        input rotation, and no gimbal-lock branch exists to say so. ``roll`` and
        ``yaw`` come from ``atan2`` of two quantities that cancel to nothing
        there, so the triple that comes back is decided by rounding: it varies
        between dtypes, between PyTorch versions, and under a one-ulp change of
        the input pitch, and no specific triple is quoted here for that reason.
        What is stable is that ``pitch`` lands within about ``sqrt(eps)`` of
        ``±pi/2`` — the ``asin`` argument rounds to within an ulp of ``1``, and
        ``asin(1 - d)`` is ``pi/2 - sqrt(2d)`` — and that the reconstructed
        rotation is far from the input. Whether ``±pi/2`` is reached exactly
        depends on dtype and build: at ``float64`` it is exact here, while the
        ``float32`` round trip of ``(0.1, pi/2, 0.2)`` returns pitch
        ``1.570451``, ``3.45e-4`` *below* ``float32``'s ``pi/2`` — an exact
        ``pitch == pi/2`` check never fires there. Random ``(roll, yaw)``
        at ``pitch = +pi/2`` fail this way, while ``|pitch| < pi/4`` round trips
        to rounding. The rotation does survive on the diagonal
        ``roll = yaw`` at ``+pi/2`` (and ``roll = -yaw`` at ``-pi/2``), where
        random draws round trip to within a few parts in ``1e8``, though
        ``roll`` and ``yaw`` are still not returned individually. Tracked in
        `#3950 <https://github.com/kornia/kornia/issues/3950>`_.

    .. warning::
        The input is **not** normalised, so a non-unit quaternion silently gives
        a wrong triple: for the ``q`` of ``(0.3, 0.7, 1.1)``, passing ``2 * q``
        returns ``[1.6560585860248003, 1.5707963267948966,
        2.1048169977173687]``. The middle value is exactly ``pi/2`` — the
        over-scaled input saturates the ``asin`` and is reported as
        gimbal-locked. Tracked in
        `#3953 <https://github.com/kornia/kornia/issues/3953>`_.

    Args:
        w: quaternion :math:`q_w` coefficient.
        x: quaternion :math:`q_x` coefficient.
        y: quaternion :math:`q_y` coefficient.
        z: quaternion :math:`q_z` coefficient.

    Return:
        A tuple with euler angles`roll`, `pitch`, `yaw`.

    """
    KORNIA_CHECK(w.shape == x.shape)
    KORNIA_CHECK(x.shape == y.shape)
    KORNIA_CHECK(y.shape == z.shape)

    yy = y * y

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + yy)
    roll = sinr_cosp.atan2(cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = sinp.clamp(min=-1.0, max=1.0)
    pitch = sinp.asin()

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (yy + z * z)
    yaw = siny_cosp.atan2(cosy_cosp)

    return roll, pitch, yaw


def quaternion_from_euler(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert Euler angles to quaternion coefficients.

    Convention:
        - ``roll``, ``pitch`` and ``yaw`` are in **radians** and turn about
          ``x``, ``y`` and ``z`` respectively; the rotation they compose to is
          ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` — extrinsic X-Y-Z about the
          fixed axes, equivalently intrinsic Z-Y'-X''. The order matters: at
          ``(0.3, 0.7, 1.1)`` the returned quaternion's matrix differs from
          ``Rx @ Ry @ Rz`` by ``0.64``, from ``Ry @ Rz @ Rx`` by ``0.55`` and
          from ``Rx @ Rz @ Ry`` by ``0.25``
        - the three arguments must have exactly equal shapes; broadcastable
          shapes such as ``(2,)`` and ``(1,)`` raise
          ``BaseError: Validation condition failed``
        - the return is a tuple of **four separate tensors** ``(w, x, y, z)``,
          not a :math:`(*, 4)` tensor: ``(0.3, 0.7, 1.1)`` gives
          ``[0.8186, -0.0575, 0.3624, 0.4418]``

    Args:
        roll: the roll euler angle.
        pitch: the pitch euler angle.
        yaw: the yaw euler angle.

    Return:
        A tuple with quaternion coefficients in order of `wxyz`.

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
    r"""Map pixel coordinates so that the first and last pixel of each axis become -1 and 1.

    Convention:
        - ``pixel_coordinates`` is :math:`(*, 2)` in ``(x, y)`` order: ``x``
          indexes columns and is scaled by ``width``, ``y`` indexes rows and is
          scaled by ``height``. The positional argument order is the other way
          round — ``(pixel_coordinates, height, width)``
        - the mapping is **corner-aligned**,
          ``x_norm = 2 * x / (width - 1) - 1``: for ``width = 4`` the columns
          ``0``, ``1``, ``3`` map to ``-1``, ``-1/3``, ``+1``. This is the
          ``align_corners=True`` convention;
          :func:`torch.nn.functional.grid_sample` at its default
          ``align_corners=False`` would instead place the first and last of
          those values at pixels ``-0.5`` and ``3.5`` — half a pixel outside
          the image on each side — so ``align_corners=True`` must be passed
          explicitly when feeding this output to it. Note that kornia's own
          :func:`~kornia.geometry.transform.remap` resolves
          ``align_corners=None`` to ``False``
        - the output is **not** clamped: coordinates outside the image
          extrapolate linearly past ``[-1, 1]``, as the example below shows

    .. warning::
        ``height`` and ``width`` are not validated. For a degenerate size (``1``,
        ``0`` or negative) the denominator ``size - 1`` is clamped up to ``eps``,
        so
        ``normalize_pixel_coordinates(torch.tensor([[1., 1.]], dtype=torch.float64), 1, 1)``
        silently returns ``[[199999999.0, 199999999.0]]`` instead of raising.
        Tracked in `#3940 <https://github.com/kornia/kornia/issues/3940>`_.

    Args:
        pixel_coordinates: the grid with pixel coordinates. Shape can be :math:`(*, 2)`.
        height: the maximum height in the y-axis.
        width: the maximum width in the x-axis.
        eps: safe division by zero.

    Return:
        the normalized pixel coordinates with shape :math:`(*, 2)`.

    Examples:
        >>> coords = torch.tensor([[50., 100.]])
        >>> normalize_pixel_coordinates(coords, 100, 50)
        tensor([[1.0408, 1.0202]])

    """
    if pixel_coordinates.shape[-1] != 2:
        raise ValueError(f"Input pixel_coordinates must be of shape (*, 2). Got {pixel_coordinates.shape}")

    # compute normalization factor
    hw: torch.Tensor = torch.stack(
        [
            torch.tensor(width, device=pixel_coordinates.device, dtype=pixel_coordinates.dtype),
            torch.tensor(height, device=pixel_coordinates.device, dtype=pixel_coordinates.dtype),
        ]
    )

    factor: torch.Tensor = torch.tensor(2.0, device=pixel_coordinates.device, dtype=pixel_coordinates.dtype) / (
        hw - 1
    ).clamp(eps)

    return factor * pixel_coordinates - 1


def denormalize_pixel_coordinates(
    pixel_coordinates: torch.Tensor, height: int, width: int, eps: float = 1e-8
) -> torch.Tensor:
    r"""Denormalize pixel coordinates.

    The input is assumed to be -1 if on extreme left, 1 if on extreme right (x = w-1).

    Convention:
        - the inverse of
          :func:`~kornia.geometry.conversions.normalize_pixel_coordinates`,
          ``x = (width - 1) * (x_norm + 1) / 2``, with the same ``(x, y)``
          component order and the same ``(pixel_coordinates, height, width)``
          positional order

    .. warning::
        For a degenerate ``height``/``width`` (``1``, ``0`` or negative) the
        clamped denominator collapses that component instead of exploding it:
        ``denormalize_pixel_coordinates(torch.tensor([[0., 0.]], dtype=torch.float64), 4, 1)``
        returns ``[[5e-09, 1.5]]``. This clamp is the exact reciprocal of the one
        in :func:`~kornia.geometry.conversions.normalize_pixel_coordinates`, so a
        normalize-then-denormalize round trip returns the input as long as the
        normalized value is finite — in ``float16`` ``eps`` underflows, the
        normalized component is ``inf`` and the round trip returns ``nan``; it
        is a single call on its own that is wrong. Tracked in
        `#3940 <https://github.com/kornia/kornia/issues/3940>`_.

    Args:
        pixel_coordinates: the normalized grid coordinates. Shape can be :math:`(*, 2)`.
        height: the maximum height in the y-axis.
        width: the maximum width in the x-axis.
        eps: safe division by zero.

    Return:
        the denormalized pixel coordinates with shape :math:`(*, 2)`.

    Examples:
        >>> coords = torch.tensor([[-1., -1.]])
        >>> denormalize_pixel_coordinates(coords, 100, 50)
        tensor([[0., 0.]])

    """
    if pixel_coordinates.shape[-1] != 2:
        raise ValueError(f"Input pixel_coordinates must be of shape (*, 2). Got {pixel_coordinates.shape}")
    # compute normalization factor
    hw: torch.Tensor = (
        torch.stack([torch.tensor(width), torch.tensor(height)])
        .to(pixel_coordinates.device)
        .to(pixel_coordinates.dtype)
    )

    factor: torch.Tensor = torch.tensor(2.0) / (hw - 1).clamp(eps)

    return torch.tensor(1.0) / factor * (pixel_coordinates + 1)


def normalize_pixel_coordinates3d(
    pixel_coordinates: torch.Tensor, depth: int, height: int, width: int, eps: float = 1e-8
) -> torch.Tensor:
    r"""Map 3d pixel coordinates so that the first and last sample of each axis become -1 and 1.

    Convention:
        - ``pixel_coordinates`` is :math:`(*, 3)` in ``(d, x, y)`` order —
          **depth first, then x, then y**, *not* ``(x, y, z)``. This is the
          order :func:`~kornia.geometry.grid.create_meshgrid3d` produces. The
          three components are scaled by ``depth - 1``, ``width - 1`` and
          ``height - 1`` respectively, so with ``depth=3, height=5, width=9``
          the coordinate ``[2., 8., 4.]`` maps to ``[1., 1., 1.]``
        - the positional argument order is
          ``(pixel_coordinates, depth, height, width)``
        - corner-aligned and unclamped in each axis exactly as in
          :func:`~kornia.geometry.conversions.normalize_pixel_coordinates`

    .. warning::
        ``depth``, ``height`` and ``width`` are not validated; a degenerate size
        (``1``, ``0`` or negative) clamps that axis' denominator up to ``eps``
        and blows the corresponding component up by a factor of ``2e8``
        (``inf`` in ``float16``, where ``eps`` underflows). Tracked
        in `#3940 <https://github.com/kornia/kornia/issues/3940>`_.

    Args:
        pixel_coordinates: the grid with pixel coordinates. Shape can be :math:`(*, 3)`.
        depth: the maximum depth in the z-axis.
        height: the maximum height in the y-axis.
        width: the maximum width in the x-axis.
        eps: safe division by zero.

    Return:
        the normalized pixel coordinates.

    """
    if pixel_coordinates.shape[-1] != 3:
        raise ValueError(f"Input pixel_coordinates must be of shape (*, 3). Got {pixel_coordinates.shape}")
    # compute normalization factor
    dhw: torch.Tensor = (
        torch.stack([torch.tensor(depth), torch.tensor(width), torch.tensor(height)])
        .to(pixel_coordinates.device)
        .to(pixel_coordinates.dtype)
    )

    factor: torch.Tensor = torch.tensor(2.0) / (dhw - 1).clamp(eps)

    return factor * pixel_coordinates - 1


def denormalize_pixel_coordinates3d(
    pixel_coordinates: torch.Tensor, depth: int, height: int, width: int, eps: float = 1e-8
) -> torch.Tensor:
    r"""Denormalize 3d pixel coordinates.

    Convention:
        - the inverse of
          :func:`~kornia.geometry.conversions.normalize_pixel_coordinates3d`,
          with the same ``(d, x, y)`` component order and the same
          ``(pixel_coordinates, depth, height, width)`` positional order: with
          ``depth=3, height=5, width=9`` the input ``[0., 0., 0.]`` maps back to
          ``[1., 4., 2.]``

    .. warning::
        For a degenerate ``depth``/``height``/``width`` (``1``, ``0`` or
        negative) the clamped denominator scales that component by ``5e-09``
        (``0`` in ``float16``, where ``eps`` underflows) instead of by
        ``(size - 1) / 2`` — the same reciprocal-clamp behavior as
        :func:`~kornia.geometry.conversions.denormalize_pixel_coordinates`,
        whose warning walks through the round trip. Tracked in
        `#3940 <https://github.com/kornia/kornia/issues/3940>`_.

    Args:
        pixel_coordinates: the normalized grid coordinates. Shape can be :math:`(*, 3)`.
        depth: the maximum depth in the z-axis.
        height: the maximum height in the y-axis.
        width: the maximum width in the x-axis.
        eps: safe division by zero.

    Return:
        the denormalized pixel coordinates.

    """
    if pixel_coordinates.shape[-1] != 3:
        raise ValueError(f"Input pixel_coordinates must be of shape (*, 3). Got {pixel_coordinates.shape}")
    # compute normalization factor
    dhw: torch.Tensor = (
        torch.stack([torch.tensor(depth), torch.tensor(width), torch.tensor(height)])
        .to(pixel_coordinates.device)
        .to(pixel_coordinates.dtype)
    )

    factor: torch.Tensor = torch.tensor(2.0) / (dhw - 1).clamp(eps)

    return torch.tensor(1.0) / factor * (pixel_coordinates + 1)


def angle_to_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    r"""Create a rotation matrix out of angles in degrees.

    Convention:
        - ``angle`` is in **degrees**, shape :math:`(*)` in and
          :math:`(*, 2, 2)` out
        - the matrix is ``[[cos, sin], [-sin, cos]]`` with ``det = +1``; for
          ``angle = 30`` it is ``[[0.8660, 0.5000], [-0.5000, 0.8660]]``
        - applied on the left to a column vector ``(x, y)``, ``angle = 30`` maps
          ``(1., 0.)`` to ``(0.8660, -0.5000)`` — counter-clockwise **as
          displayed** under kornia's y-down image axes. In the raw coordinate
          plane this is the opposite sense to
          :func:`~kornia.geometry.conversions.cart2pol`, whose Convention block
          spells out the modulo-:math:`2\pi` relation between the two ops

    .. warning::
        The degrees-to-radians step is
        :func:`~kornia.geometry.conversions.deg2rad`, so it inherits both
        defects of :func:`~kornia.geometry.conversions.rad2deg` — the float32
        ``kornia.constants.pi`` and the cast to the input dtype. See its
        warning and `#3937 <https://github.com/kornia/kornia/issues/3937>`_.

    Args:
        angle: tensor of angles in degrees, any shape :math:`(*)`.

    Returns:
        tensor of rotation matrices with shape :math:`(*, 2, 2)`.

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
    r"""Normalize a given homography in pixels to [-1, 1].

    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          normalized. :math:`(B, 3, 3)`
        dsize_src: size of the source image (height, width).
        dsize_dst: size of the destination image (height, width).
        align_corners: which :py:func:`torch.nn.functional.grid_sample` convention the
          normalized :math:`[-1, 1]` coordinates follow, forwarded to
          :func:`normal_transform_pixel`. Must match the ``align_corners`` of the
          ``grid_sample`` call that ultimately consumes the result, otherwise the warp
          picks up a spurious sub-pixel scale and shift.

    Returns:
        the normalized homography of shape :math:`(B, 3, 3)`.

    """
    if not isinstance(dst_pix_trans_src_pix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(dst_pix_trans_src_pix)}")

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {dst_pix_trans_src_pix.shape}")

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w, align_corners=align_corners).to(
        dst_pix_trans_src_pix
    )

    # Closed-form 3x3 inverse of the (well-conditioned) pixel-normalization matrix: cusolver-free,
    # so homography normalization runs on the Jetson wheel where ``torch.linalg.inv`` dlopen-fails.
    src_pix_trans_src_norm = _inverse_3x3_closed_form(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w, align_corners=align_corners).to(
        dst_pix_trans_src_pix
    )

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
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        height: image height.
        width: image width.
        eps: epsilon to prevent divide-by-zero errors. Only used by the
          ``align_corners=True`` mapping, which is singular for a size of 1.
        device: device to place the result on.
        dtype: dtype of the result.
        align_corners: which :py:func:`torch.nn.functional.grid_sample` convention to
          normalize to. ``True`` maps pixel centers :math:`[0, size-1]` to
          :math:`[-1, 1]`; ``False`` uses the half-pixel mapping
          :math:`x_{norm} = (2x + 1) / W - 1`, where :math:`\pm 1` are the outer pixel
          *edges*.

    Returns:
        normalized transform with shape :math:`(1, 3, 3)`.

    """
    if align_corners:
        # prevent divide by zero bugs
        width_denom: float = eps if width == 1 else width - 1.0
        height_denom: float = eps if height == 1 else height - 1.0

        sx: float = 2.0 / width_denom
        sy: float = 2.0 / height_denom
        tx: float = -1.0
        ty: float = -1.0
    else:
        # Half-pixel mapping; well defined for a size of 1 (which lands on the pixel center, 0).
        sx = 2.0 / width
        sy = 2.0 / height
        tx = 1.0 / width - 1.0
        ty = 1.0 / height - 1.0

    # Construct the matrix in one shot (no in-place mutation).
    tr_mat = torch.tensor(
        [[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )  # 3x3

    return tr_mat.unsqueeze(0)  # 1x3x3


def normal_transform_pixel3d(
    depth: int,
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        depth: image depth.
        height: image height.
        width: image width.
        eps: epsilon to prevent divide-by-zero errors
        device: device to place the result on.
        dtype: dtype of the result.

    Returns:
        normalized transform with shape :math:`(1, 4, 4)`.

    """
    tr_mat = torch.tensor(
        [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )  # 4x4

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0
    depth_denom: float = eps if depth == 1 else depth - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom
    tr_mat[2, 2] = tr_mat[2, 2] * 2.0 / depth_denom

    return tr_mat.unsqueeze(0)  # 1x4x4


def denormalize_homography(
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: tuple[int, int],
    dsize_dst: tuple[int, int],
    align_corners: bool = True,
) -> torch.Tensor:
    r"""De-normalize a given homography in pixels from [-1, 1] to actual height and width.

    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          denormalized. :math:`(B, 3, 3)`
        dsize_src: size of the source image (height, width).
        dsize_dst: size of the destination image (height, width).
        align_corners: which :py:func:`torch.nn.functional.grid_sample` convention the
          incoming normalized coordinates follow. Must match the value used to produce
          them, so that this function inverts :func:`normalize_homography`.

    Returns:
        the denormalized homography of shape :math:`(B, 3, 3)`.

    """
    if not isinstance(dst_pix_trans_src_pix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(dst_pix_trans_src_pix)}")

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {dst_pix_trans_src_pix.shape}")

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w, align_corners=align_corners).to(
        dst_pix_trans_src_pix
    )

    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w, align_corners=align_corners).to(
        dst_pix_trans_src_pix
    )
    dst_denorm_trans_dst_pix = _torch_inverse_cast(dst_norm_trans_dst_pix)
    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_denorm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_norm_trans_src_pix)
    return dst_norm_trans_src_norm


def normalize_homography3d(
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: tuple[int, int, int], dsize_dst: tuple[int, int, int]
) -> torch.Tensor:
    r"""Normalize a given homography in pixels to [-1, 1].

    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          normalized. :math:`(B, 4, 4)`
        dsize_src: size of the source image (depth, height, width).
        dsize_dst: size of the destination image (depth, height, width).

    Returns:
        the normalized homography.

    Shape:
        Output: :math:`(B, 4, 4)`

    """
    if not isinstance(dst_pix_trans_src_pix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(dst_pix_trans_src_pix)}")

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (4, 4)):
        raise ValueError(f"Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {dst_pix_trans_src_pix.shape}")

    # source and destination sizes
    src_d, src_h, src_w = dsize_src
    dst_d, dst_h, dst_w = dsize_dst
    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel3d(src_d, src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel3d(dst_d, dst_h, dst_w).to(dst_pix_trans_src_pix)
    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def normalize_points_with_intrinsics(point_2d: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
    """Normalize points with intrinsics. Useful for conversion of keypoints to be used with essential matrix.

    Convention:
        - ``point_2d`` is :math:`(*, 2)` in pixel ``(u, v)`` order and
          ``camera_matrix`` is a row-major pinhole :math:`(*, 3, 3)` with
          ``fx = K[0, 0]``, ``fy = K[1, 1]``, ``cx = K[0, 2]``, ``cy = K[1, 2]``
        - the output is the calibrated camera coordinate
          ``x = (u - cx) / fx``, ``y = (v - cy) / fy``: with
          ``K = [[100., 0., 320.], [0., 200., 240.], [0., 0., 1.]]`` the pixel
          ``(420., 440.)`` maps to ``(1., 1.)``
        - the skew entry ``K[0, 1]`` is **ignored**; only the two diagonal
          entries and the ``[:2, 2]`` column are read

    Args:
        point_2d: tensor containing the 2d points in the image pixel coordinates. The shape of the tensor can be
                  :math:`(*, 2)`.
        camera_matrix: tensor containing the intrinsics camera matrix. The tensor shape must be :math:`(*, 3, 3)`.

    Returns:
        tensor of normalized camera coordinates :math:`(x, y)` with shape :math:`(*, 2)`.

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
    xy = (point_2d - cxcy) / fxfy
    return xy


def denormalize_points_with_intrinsics(point_2d_norm: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
    """Denormalize points with intrinsics. Useful for converting normalized camera points back to pixels.

    Convention:
        - the inverse of
          :func:`~kornia.geometry.conversions.normalize_points_with_intrinsics`,
          which documents the ``K`` layout: ``u = x * fx + cx`` and
          ``v = y * fy + cy``, and the skew entry ``K[0, 1]`` is ignored here
          too

    Args:
        point_2d_norm: tensor containing the 2d points in normalized camera coordinates. The shape of the tensor can
                       be :math:`(*, 2)`.
        camera_matrix: tensor containing the intrinsics camera matrix. The tensor shape must be :math:`(*, 3, 3)`.

    Returns:
        tensor of :math:`(u, v)` pixel coordinates with shape :math:`(*, 2)`.

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
    r"""Combine 3x3 rotation matrix R and 1x3 translation vector t into 4x4 extrinsics.

    Args:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.

    Returns:
        the extrinsics :math:`(B, 4, 4)`.

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
    Rt = torch.cat([R, t], dim=2)
    return convert_affinematrix_to_homography3d(Rt)


def matrix4x4_to_Rt(extrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert 4x4 extrinsics into 3x3 rotation matrix R and 1x3 translation vector ts.

    Args:
        extrinsics: pose matrix :math:`(B, 4, 4)`.

    Returns:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.

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


def camtoworld_graphics_to_vision_4x4(extrinsics_graphics: torch.Tensor) -> torch.Tensor:
    r"""Convert graphics coordinate frame (e.g. OpenGL) to vision coordinate frame (e.g. OpenCV.).

    I.e. flips y and z axis. Graphics convention: [+x, +y, +z] == [right, up, backwards].
    Vision convention: [+x, +y, +z] == [right, down, forwards].

    Args:
        extrinsics_graphics: pose matrix :math:`(B, 4, 4)`.

    Returns:
        extrinsics: pose matrix :math:`(B, 4, 4)`.

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
    return extrinsics_graphics @ invert_yz


def camtoworld_graphics_to_vision_Rt(R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert graphics coordinate frame (e.g. OpenGL) to vision coordinate frame (e.g. OpenCV.).

    I.e. flips y and z axis. Graphics convention: [+x, +y, +z] == [right, up, backwards].
    Vision convention: [+x, +y, +z] == [right, down, forwards].

    Args:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.

    Returns:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.

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
    mat4x4 = camtoworld_graphics_to_vision_4x4(Rt_to_matrix4x4(R, t))
    return matrix4x4_to_Rt(mat4x4)


def camtoworld_vision_to_graphics_4x4(extrinsics_vision: torch.Tensor) -> torch.Tensor:
    r"""Convert vision coordinate frame (e.g. OpenCV) to graphics coordinate frame (e.g. OpenGK.).

    I.e. flips y and z axis Graphics convention: [+x, +y, +z] == [right, up, backwards].
    Vision convention: [+x, +y, +z] == [right, down, forwards].

    Args:
        extrinsics_vision: pose matrix :math:`(B, 4, 4)`.

    Returns:
        extrinsics: pose matrix :math:`(B, 4, 4)`.

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
    return extrinsics_vision @ invert_yz


def camtoworld_vision_to_graphics_Rt(R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert graphics coordinate frame (e.g. OpenGL) to vision coordinate frame (e.g. OpenCV.).

    I.e. flips y and z axis. Graphics convention: [+x, +y, +z] == [right, up, backwards].
    Vision convention: [+x, +y, +z] == [right, down, forwards]

    Args:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.

    Returns:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.

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
    mat4x4 = camtoworld_vision_to_graphics_4x4(Rt_to_matrix4x4(R, t))
    return matrix4x4_to_Rt(mat4x4)


def camtoworld_to_worldtocam_Rt(R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert camtoworld to worldtocam frame used in Colmap.

    See
    long-url: https://colmap.github.io/format.html#output-format

    Args:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.

    Returns:
        Rinv: Rotation matrix, :math:`(B, 3, 3).`
        tinv: Translation matrix :math:`(B, 3, 1)`.

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

    R_inv = R.transpose(1, 2)
    new_t: torch.Tensor = -R_inv @ t

    return (R_inv, new_t)


def worldtocam_to_camtoworld_Rt(R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert worldtocam frame used in Colmap to camtoworld.

    Args:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.

    Returns:
        Rinv: Rotation matrix, :math:`(B, 3, 3).`
        tinv: Translation matrix :math:`(B, 3, 1)`.

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

    R_inv = R.transpose(1, 2)
    new_t: torch.Tensor = -R_inv @ t

    return (R_inv, new_t)


def ARKitQTVecs_to_ColmapQTVecs(qvec: torch.Tensor, tvec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert output of Apple ARKit screen pose to the camera-to-world transformation, expected by Colmap.

    Both poses in quaternion representation.

    Args:
        qvec: ARKit rotation quaternion :math:`(B, 4)`, [w, x, y, z] format.
        tvec: translation vector :math:`(B, 3, 1)`, [x, y, z]

    Returns:
        qvec: Colmap rotation quaternion :math:`(B, 4)`, [w, x, y, z] format.
        tvec: translation vector :math:`(B, 3, 1)`, [x, y, z]

    Example:
        >>> q, t = torch.tensor([0, 1, 0, 1.])[None], torch.ones(3).reshape(1, 3, 1)
        >>> ARKitQTVecs_to_ColmapQTVecs(q, t)
        (tensor([[0.7071, 0.0000, 0.7071, 0.0000]]), tensor([[[-1.0000],
                 [-1.0000],
                 [ 1.0000]]]))

    """
    # ToDo:  integrate QuaterniaonAPI

    Rcg = quaternion_to_rotation_matrix(qvec)
    Rcv, Tcv = camtoworld_graphics_to_vision_Rt(Rcg, tvec)
    R_colmap, t_colmap = camtoworld_to_worldtocam_Rt(Rcv, Tcv)
    t_colmap = t_colmap.reshape(-1, 3, 1)
    q_colmap = rotation_matrix_to_quaternion(R_colmap.contiguous())
    return q_colmap, t_colmap


def vector_to_skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:
    r"""Convert a vector to a skew symmetric matrix.

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
        - accepts :math:`(3,)` or :math:`(B, 3)` only; any higher rank raises
          ``ValueError``

    Args:
        vec: tensor of shape :math:`(3,)` or :math:`(B, 3)`.

    Returns:
        tensor of shape :math:`(3, 3)` or :math:`(B, 3, 3)` respectively.

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
    skew_symmetric_matrix = torch.stack(
        [
            torch.stack([zeros, -v3, v2], dim=-1),
            torch.stack([v3, zeros, -v1], dim=-1),
            torch.stack([-v2, v1, zeros], dim=-1),
        ],
        dim=-2,
    )
    return skew_symmetric_matrix
