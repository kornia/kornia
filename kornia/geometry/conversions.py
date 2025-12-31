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
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.utils import deprecated
from kornia.utils.helpers import _torch_inverse_cast

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
    """Convert cartesian coordinates to polar coordinates.

    Args:
        x: torch.Tensor of arbitrary shape.
        y: torch.Tensor of same arbitrary shape.
        eps: To avoid division by zero.

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

    Args:
        points: the points to be transformed of shape :math:`(B, N, D)`.
        eps: to avoid division by zero.

    Returns:
        the points in Euclidean space :math:`(B, N, D-1)`.

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

    Args:
        rotation_matrix: rotation matrix of shape :math:`(N, 3, 3)`.

    Returns:
        Rodrigues vector transformation of shape :math:`(N, 3)`.

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

    Args:
        rotation_matrix: the rotation matrix to convert with shape :math:`(*, 3, 3)`.
        eps: small value to avoid zero division.

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

    Args:
        quaternion: a tensor containing a quaternion to be normalized.
          The tensor can be of shape :math:`(*, 4)`.
        eps: small value to avoid division by zero.

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
    q1: torch.Tensor = torch.tensor([])
    q2: torch.Tensor = torch.tensor([])
    q3: torch.Tensor = torch.tensor([])
    cos_theta: torch.Tensor = torch.tensor([])

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

    The quaternion should be in (w, x, y, z) format.

    Args:
        quaternion: a tensor containing a quaternion to be converted.
          The tensor can be of shape :math:`(*, 3)`.
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
    quaternion_exp: torch.Tensor = torch.tensor([])
    quaternion_exp = torch.cat((quaternion_scalar, quaternion_vector), dim=-1)

    return quaternion_exp


def quaternion_exp_to_log(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    r"""Apply the log map to a quaternion.

    The quaternion should be in (w, x, y, z) format.

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
    quaternion_vector: torch.Tensor = torch.tensor([])
    quaternion_scalar: torch.Tensor = torch.tensor([])

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

    Returned angles are in radians in XYZ convention.

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

    Euler angles are assumed to be in radians in XYZ convention.

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
    r"""Normalize pixel coordinates between -1 and 1.

    Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).

    Args:
        pixel_coordinates: the grid with pixel coordinates. Shape can be :math:`(*, 2)`.
        width: the maximum width in the x-axis.
        height: the maximum height in the y-axis.
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

    Args:
        pixel_coordinates: the normalized grid coordinates. Shape can be :math:`(*, 2)`.
        width: the maximum width in the x-axis.
        height: the maximum height in the y-axis.
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
    r"""Normalize pixel coordinates between -1 and 1.

    Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).

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
    r"""Denormalize pixel coordinates.

    The input is assumed to be -1 if on extreme left, 1 if on extreme right (x = w-1).

    Args:
        pixel_coordinates: the normalized grid coordinates. Shape can be :math:`(*, 3)`.
        depth: the maximum depth in the x-axis.
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
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: tuple[int, int], dsize_dst: tuple[int, int]
) -> torch.Tensor:
    r"""Normalize a given homography in pixels to [-1, 1].

    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          normalized. :math:`(B, 3, 3)`
        dsize_src: size of the source image (height, width).
        dsize_dst: size of the destination image (height, width).

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
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        height: image height.
        width: image width.
        eps: epsilon to prevent divide-by-zero errors
        device: device to place the result on.
        dtype: dtype of the result.

    Returns:
        normalized transform with shape :math:`(1, 3, 3)`.

    """
    tr_mat = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

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
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: tuple[int, int], dsize_dst: tuple[int, int]
) -> torch.Tensor:
    r"""De-normalize a given homography in pixels from [-1, 1] to actual height and width.

    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          denormalized. :math:`(B, 3, 3)`
        dsize_src: size of the source image (height, width).
        dsize_dst: size of the destination image (height, width).

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
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(dst_pix_trans_src_pix)

    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(dst_pix_trans_src_pix)
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

    Args:
        point_2d: tensor containing the 2d points in the image pixel coordinates. The shape of the tensor can be
                  :math:`(*, 2)`.
        camera_matrix: tensor containing the intrinsics camera matrix. The tensor shape must be :math:`(*, 3, 3)`.

    Returns:
        tensor of (u, v) cam coordinates with shape :math:`(*, 2)`.

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
    """Normalize points with intrinsics. Useful for conversion of keypoints to be used with essential matrix.

    Args:
        point_2d_norm: tensor containing the 2d points in the image pixel coordinates. The shape of the tensor can be
                       :math:`(*, 2)`.
        camera_matrix: tensor containing the intrinsics camera matrix. The tensor shape must be :math:`(*, 3, 3)`.

    Returns:
        tensor of (u, v) cam coordinates with shape :math:`(*, 2)`.

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

    # unpack coordinates
    x_coord: torch.Tensor = point_2d_norm[..., 0]
    y_coord: torch.Tensor = point_2d_norm[..., 1]

    # unpack intrinsics
    fx: torch.Tensor = camera_matrix[..., 0, 0]
    fy: torch.Tensor = camera_matrix[..., 1, 1]
    cx: torch.Tensor = camera_matrix[..., 0, 2]
    cy: torch.Tensor = camera_matrix[..., 1, 2]

    if len(cx.shape) < len(x_coord.shape):  # broadcast intrinsics
        cx, cy, fx, fy = cx.unsqueeze(-1), cy.unsqueeze(-1), fx.unsqueeze(-1), fy.unsqueeze(-1)

    # apply intrinsics ans return
    u_coord: torch.Tensor = x_coord * fx + cx
    v_coord: torch.Tensor = y_coord * fy + cy

    return torch.stack([u_coord, v_coord], dim=-1)


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

    Args:
        vec: tensor of shape :math:`(B, 3)`.

    Returns:
        tensor of shape :math:`(B, 3, 3)`.

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
