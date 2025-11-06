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

from kornia.core import Tensor, pad
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.geometry.conversions import convert_points_from_homogeneous

__all__ = [
    "batched_dot_product",
    "batched_squared_norm",
    "compose_transformations",
    "euclidean_distance",
    "inverse_transformation",
    "point_line_distance",
    "relative_transformation",
    "squared_norm",
    "transform_points",
]


def compose_transformations(trans_01: Tensor, trans_12: Tensor) -> Tensor:
    r"""Compose two homogeneous transformations.

    .. math::
        T_0^{2} = \begin{bmatrix} R_0^1 R_1^{2} & R_0^{1} t_1^{2} + t_0^{1} \
        \\mathbf{0} & 1\end{bmatrix}

    Args:
        trans_01: tensor with the homogeneous transformation from
          a reference frame 1 respect to a frame 0. The tensor has must have a
          shape of :math:`(N, 4, 4)` or :math:`(4, 4)`.
        trans_12: tensor with the homogeneous transformation from
          a reference frame 2 respect to a frame 1. The tensor has must have a
          shape of :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        the transformation between the two frames with shape :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_12 = torch.eye(4)  # 4x4
        >>> trans_02 = compose_transformations(trans_01, trans_12)  # 4x4

    """
    KORNIA_CHECK_IS_TENSOR(trans_01)
    KORNIA_CHECK_IS_TENSOR(trans_12)

    if not ((trans_01.dim() in (2, 3)) and (trans_01.shape[-2:] == (4, 4))):
        raise ValueError(f"Input trans_01 must be a of the shape Nx4x4 or 4x4. Got {trans_01.shape}")

    if not ((trans_12.dim() in (2, 3)) and (trans_12.shape[-2:] == (4, 4))):
        raise ValueError(f"Input trans_12 must be a of the shape Nx4x4 or 4x4. Got {trans_12.shape}")

    if trans_01.dim() != trans_12.dim():
        raise ValueError(f"Input number of dims must match. Got {trans_01.dim()} and {trans_12.dim()}")

    # unpack input data
    rmat_01 = trans_01[..., :3, :3]
    rmat_12 = trans_12[..., :3, :3]
    tvec_01 = trans_01[..., :3, 3:]
    tvec_12 = trans_12[..., :3, 3:]

    # compute the actual transforms composition
    rmat_02 = torch.matmul(rmat_01, rmat_12)
    tvec_02 = torch.matmul(rmat_01, tvec_12) + tvec_01

    trans_02 = trans_01.new_zeros(trans_01.shape)
    trans_02[..., :3, :3] = rmat_02
    trans_02[..., :3, 3:] = tvec_02
    trans_02[..., 3, 3] = 1.0
    return trans_02


def inverse_transformation(trans_12: Tensor) -> Tensor:
    r"""Invert a 4x4 homogeneous transformation.

     :math:`T_1^{2} = \begin{bmatrix} R_1 & t_1 \\ \mathbf{0} & 1 \end{bmatrix}`

    The inverse transformation is computed as follows:

    .. math::

        T_2^{1} = (T_1^{2})^{-1} = \begin{bmatrix} R_1^T & -R_1^T t_1 \\
        \mathbf{0} & 1\end{bmatrix}

    Args:
        trans_12: transformation tensor of shape :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        tensor with inverted transformations with shape :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Example:
        >>> trans_12 = torch.rand(1, 4, 4)  # Nx4x4
        >>> trans_21 = inverse_transformation(trans_12)  # Nx4x4

    """
    KORNIA_CHECK_IS_TENSOR(trans_12)

    if not ((trans_12.dim() in (2, 3)) and (trans_12.shape[-2:] == (4, 4))):
        raise ValueError(f"Input size must be a Nx4x4 or 4x4. Got {trans_12.shape}")
    # unpack input tensor
    rmat_12 = trans_12[..., :3, :3]  # Nx3x3 or 3x3
    tvec_12 = trans_12[..., :3, 3:4]  # Nx3x1 or 3x1

    # compute the actual inverse
    rmat_21 = rmat_12.transpose(-1, -2)
    tvec_21 = torch.matmul(-rmat_21, tvec_12)

    # pack to output tensor
    trans_21 = trans_12.new_zeros(trans_12.shape)
    trans_21[..., :3, :3].copy_(rmat_21)
    trans_21[..., :3, 3:4].copy_(tvec_21)
    trans_21[..., 3, 3] = 1.0
    return trans_21


def relative_transformation(trans_01: Tensor, trans_02: Tensor) -> Tensor:
    r"""Compute the relative homogeneous transformation from a reference transformation.

    :math:`T_1^{0} = \begin{bmatrix} R_1 & t_1 \\ \mathbf{0} & 1 \end{bmatrix}` to destination :math:`T_2^{0} =
    \begin{bmatrix} R_2 & t_2 \\ \mathbf{0} & 1 \end{bmatrix}`.

    The relative transformation is computed as follows:

    .. math::

        T_1^{2} = (T_0^{1})^{-1} \cdot T_0^{2}

    Args:
        trans_01: reference transformation tensor of shape :math:`(N, 4, 4)` or :math:`(4, 4)`.
        trans_02: destination transformation tensor of shape :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        the relative transformation between the transformations with shape :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_02 = torch.eye(4)  # 4x4
        >>> trans_12 = relative_transformation(trans_01, trans_02)  # 4x4

    """
    KORNIA_CHECK_IS_TENSOR(trans_01)
    KORNIA_CHECK_IS_TENSOR(trans_02)
    if not ((trans_01.dim() in (2, 3)) and (trans_01.shape[-2:] == (4, 4))):
        raise ValueError(f"Input must be a of the shape Nx4x4 or 4x4. Got {trans_01.shape}")
    if not ((trans_02.dim() in (2, 3)) and (trans_02.shape[-2:] == (4, 4))):
        raise ValueError(f"Input must be a of the shape Nx4x4 or 4x4. Got {trans_02.shape}")
    if not trans_01.dim() == trans_02.dim():
        raise ValueError(f"Input number of dims must match. Got {trans_01.dim()} and {trans_02.dim()}")

    rmat_01 = trans_01[..., :3, :3]
    tvec_01 = trans_01[..., :3, 3:4]
    rmat_02 = trans_02[..., :3, :3]
    tvec_02 = trans_02[..., :3, 3:4]
    rmat_10 = rmat_01.transpose(-1, -2)
    rmat_12 = torch.matmul(rmat_10, rmat_02)
    tvec_12 = torch.matmul(rmat_10, tvec_02 - tvec_01)
    trans_12 = torch.zeros_like(trans_01)
    trans_12[..., :3, :3] = rmat_12
    trans_12[..., :3, 3:4] = tvec_12
    trans_12[..., 3, 3] = 1.0

    return trans_12


def transform_points(trans_01: Tensor, points_1: Tensor) -> Tensor:
    r"""Apply homogeneous transformations to a set of points.

    This function supports arbitrary leading batch dimensions and uses
    broadcasted ``matmul`` for speed.

    Args:
        trans_01: Transformation matrices of shape :math:`(*, D+1, D+1)`.
                  The leading batch shape must be broadcastable to the
                  leading batch shape of ``points_1``.
        points_1: Points of shape :math:`(*, N, D)`.

    Returns:
        Transformed points of shape :math:`(*, N, D)`.

    Broadcasting rules:
        Any number of leading batch dimensions is allowed. The leading batch
        shape of ``trans_01`` must be broadcastable to that of ``points_1``.
        Typical special cases that work:
            - ``trans_01`` has shape :math:`(1, D+1, D+1)` (single transform
              broadcast across all batches),
            - ``trans_01`` has no batch dims :math:`(D+1, D+1)`,
            - fully batched :math:`(M, B, D+1, D+1)` with points
              :math:`(M, B, N, D)`.

    Shape:
        - Input:  ``points_1`` :math:`(*, N, D)`, ``trans_01`` :math:`(*, D+1, D+1)`
        - Output: :math:`(*, N, D)`

    Examples:
        >>> # classic BxNxD
        >>> points_1 = torch.rand(2, 4, 3)          # (B=2, N=4, D=3)
        >>> trans_01 = torch.eye(4).expand(2, -1, -1)  # (B=2, 4, 4)
        >>> points_0 = transform_points(trans_01, points_1)

        >>> # extra leading dims: MxBxNxD with a single transform
        >>> M, B, N, D = 3, 2, 5, 2
        >>> points_1 = torch.rand(M, B, N, D)
        >>> trans_01 = torch.eye(D+1)               # ()x(D+1)x(D+1), broadcasts
        >>> points_0 = transform_points(trans_01, points_1)

        >>> # extra leading dims with fully batched transforms
        >>> trans_01 = torch.eye(D+1).expand(M, B, D+1, D+1)
        >>> points_0 = transform_points(trans_01, points_1)
    """
    KORNIA_CHECK_IS_TENSOR(trans_01)
    KORNIA_CHECK_IS_TENSOR(points_1)

    if points_1.ndim < 3:
        raise ValueError(f"`points_1` must be at least 3D (*, N, D). Got shape {points_1.shape}.")

    D = points_1.shape[-1]
    if trans_01.shape[-2:] != (D + 1, D + 1):
        raise ValueError(
            f"Last two dims of `trans_01` must be (D+1, D+1) with D={D}. "
            f"Got {trans_01.shape[-2:]} for shape {trans_01.shape}."
        )

    # Validate broadcastability of leading batch dims
    pts_batch = points_1.shape[:-2]
    trn_batch = trans_01.shape[:-2]
    try:
        _ = torch.broadcast_shapes(pts_batch, trn_batch)
    except RuntimeError as e:
        raise ValueError(
            f"Leading batch dims of `trans_01` {trn_batch} are not broadcastable "
            f"to those of `points_1` {pts_batch}."
        ) from e

    # to homogeneous: (..., N, D+1)
    points_1_h = pad(points_1, (0, 1), value=1.0)

    # transform: (..., N, D+1) @ (..., D+1, D+1) -> (..., N, D+1)
    points_0_h = torch.matmul(points_1_h, trans_01.transpose(-2, -1))

    # back to Euclidean: (..., N, D)
    points_0 = convert_points_from_homogeneous(points_0_h)
    return points_0


def point_line_distance(point: Tensor, line: Tensor, eps: float = 1e-9) -> Tensor:
    r"""Return the distance from points to lines.

    Args:
       point: (possibly homogeneous) points :math:`(*, N, 2 or 3)`.
       line: lines coefficients :math:`(a, b, c)` with shape :math:`(*, N, 3)`, where :math:`ax + by + c = 0`.
       eps: Small constant for safe sqrt.

    Returns:
        the computed distance with shape :math:`(*, N)`.

    """
    KORNIA_CHECK_IS_TENSOR(point)
    KORNIA_CHECK_IS_TENSOR(line)

    if point.shape[-1] not in (2, 3):
        raise ValueError(f"pts must be a (*, 2 or 3) tensor. Got {point.shape}")

    if line.shape[-1] != 3:
        raise ValueError(f"lines must be a (*, 3) tensor. Got {line.shape}")

    # Using in-place operations to improve performance
    numerator = line[..., 0] * point[..., 0]
    numerator += line[..., 1] * point[..., 1]
    numerator += line[..., 2]
    numerator.abs_()

    # Avoid computing norm multiple times by saving its value
    denom_norm = (line[..., 0].square() + line[..., 1].square()).sqrt()

    return numerator / (denom_norm + eps)


def batched_dot_product(x: Tensor, y: Tensor, keepdim: bool = False) -> Tensor:
    """Return a batched version of .dot()."""
    KORNIA_CHECK_SHAPE(x, ["*", "N"])
    KORNIA_CHECK_SHAPE(y, ["*", "N"])
    return (x * y).sum(-1, keepdim)


def batched_squared_norm(x: Tensor, keepdim: bool = False) -> Tensor:
    """Return the squared norm of a vector."""
    return batched_dot_product(x, x, keepdim)


def euclidean_distance(x: Tensor, y: Tensor, keepdim: bool = False, eps: float = 1e-6) -> Tensor:
    """Compute the Euclidean distance between two set of n-dimensional points.

    More: https://en.wikipedia.org/wiki/Euclidean_distance

    Args:
        x: first set of points of shape :math:`(*, N)`.
        y: second set of points of shape :math:`(*, N)`.
        keepdim: whether to keep the dimension after reduction.
        eps: small value to have numerical stability.

    """
    KORNIA_CHECK_SHAPE(x, ["*", "N"])
    KORNIA_CHECK_SHAPE(y, ["*", "N"])

    return (x - y).pow(2).sum(dim=-1, keepdim=keepdim).add_(eps).sqrt_()


# aliases
squared_norm = batched_squared_norm

# TODO:
# - project_points: from opencv
