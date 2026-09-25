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

"""Module for image projections."""

from typing import Tuple, Union

import torch
from torch.linalg import qr as linalg_qr

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.ops import eye_like, vec_like
from kornia.core.utils import _torch_svd_cast

from .numeric import cross_product_matrix


def intrinsics_like(focal: float, input: torch.Tensor) -> torch.Tensor:
    r"""Return a 3x3 intrinsics matrix per batch element of the input.

    Convention:
        - ``fx = fy = focal`` and the principal point is ``(W / 2, H / 2)`` of the :math:`(B, C, H, W)` input;
          dtype and device follow the input. ``focal`` must be positive and the input 4-D.
        - The input must be floating point, like kornia's images: an integer input raises by design rather than
          returning an integer ``K``.
        - Known defects: ``(W / 2, H / 2)`` is the half-pixel centre, not the integer-pixel centre
          ``((W - 1) / 2, (H - 1) / 2)`` of kornia's grids
          (`#4263 <https://github.com/kornia/kornia/issues/4263>`_).

    Args:
        focal: the focal length for the camera matrix.
        input: image tensor that will determine the batch size and image height
          and width. It is assumed to be a floating-point tensor in the shape of :math:`(B, C, H, W)`.

    Returns:
        The camera matrix with the shape of :math:`(B, 3, 3)`.

    """
    if len(input.shape) != 4:
        raise AssertionError(input.shape)
    if focal <= 0:
        raise AssertionError(focal)

    _, _, H, W = input.shape

    intrinsics = eye_like(3, input)
    intrinsics[..., 0, 0] *= focal
    intrinsics[..., 1, 1] *= focal
    intrinsics[..., 0, 2] += 1.0 * W / 2
    intrinsics[..., 1, 2] += 1.0 * H / 2
    return intrinsics


def random_intrinsics(low: Union[float, torch.Tensor], high: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Generate a random camera matrix based on a given uniform distribution.

    Convention:
        - ``fx``, ``fy``, ``cx`` and ``cy`` are four draws from :math:`U(low, high)` on torch's global generator.
          The bounds are scalars; dtype and device follow tensor bounds, and Python floats give the default dtype
          and device.

    Args:
        low: lower range (inclusive).
        high: upper range (exclusive).

    Returns:
        the random camera matrix with the shape of :math:`(1, 3, 3)`.

    """
    sampler = torch.distributions.Uniform(low, high)
    params = sampler.sample((4,))
    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    camera_matrix = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=fx.dtype, device=fx.device)
    return camera_matrix.unsqueeze(0)


def scale_intrinsics(camera_matrix: torch.Tensor, scale_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Scale a camera matrix containing the intrinsics.

    Applies the scaling factor to the focal length and center of projection.

    Convention:
        - Applies the rule of :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.scale` and returns a new tensor.
        - Known defects: the principal-point rule of
          :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.scale`, and the skew ``K[0, 1]``, which is left
          unscaled although resizing by ``scale_factor`` scales it
          (`#4263 <https://github.com/kornia/kornia/issues/4263>`_).

    Args:
        camera_matrix: the camera calibration matrix containing the intrinsic
          parameters. The expected shape for the tensor is :math:`(B, 3, 3)`.
        scale_factor: the scaling factor to be applied: a float, or a tensor of shape :math:`(B,)` that
          scales each batch element by its own factor.

    Returns:
        The scaled camera matrix with the same shape as the input :math:`(B, 3, 3)`.

    """
    K_scale = camera_matrix.clone()
    K_scale[..., 0, 0] *= scale_factor
    K_scale[..., 1, 1] *= scale_factor
    K_scale[..., 0, 2] *= scale_factor
    K_scale[..., 1, 2] *= scale_factor
    return K_scale


def projection_from_KRt(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""Get the projection matrix P from K, R and t.

    This function computes the product :math:`P = K [R|t]`.

    Convention:
        - ``R`` and ``t`` take world points into the camera frame, ``X_cam = R @ X + t``, and ``K`` takes the
          camera frame to pixels. ``t`` is the extrinsic translation; the camera centre is ``-R.T @ t``.
        - ``K``, ``R`` and ``t`` must have the same number of dimensions, and ``R`` and ``t`` the same batch shape;
          ``K`` broadcasts against them. The inverse is
          :func:`KRt_from_projection`, and :func:`depth_from_point` gives the depth of a point in this camera.

    Args:
       K: the camera matrix with the intrinsics with shape :math:`(*, 3, 3)`.
       R: The rotation matrix with shape :math:`(*, 3, 3)`.
       t: The translation vector with shape :math:`(*, 3, 1)`.

    Returns:
       The projection matrix P with shape :math:`(*, 3, 4)`.

    """
    KORNIA_CHECK_SHAPE(K, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(R, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(t, ["*", "3", "1"])
    if not len(K.shape) == len(R.shape) == len(t.shape):
        raise AssertionError

    Rt = torch.cat([R, t], dim=-1)  # 3x4
    Rt_h = torch.nn.functional.pad(Rt, [0, 0, 0, 1], "constant", 0.0)  # 4x4
    Rt_h[..., -1, -1] += 1.0

    K_h = torch.nn.functional.pad(K, [0, 1, 0, 1], "constant", 0.0)  # 4x4
    K_h[..., -1, -1] += 1.0

    return K @ Rt


def KRt_from_projection(P: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Decompose the Projection matrix into ``K``, ``R`` and ``t`` with :math:`P = K [R|t]`.

    Convention:
        - Inverse of :func:`projection_from_KRt`: returns ``K`` (upper triangular, positive diagonal), a
          rotation ``R`` (``det R = 1``) and the translation ``t``, not the camera centre;
          :ref:`Two-view geometry <two-view-conventions>` maps this onto OpenCV.
        - ``P`` is defined up to a nonzero scale, sign included: ``s * P`` gives the same ``R`` and ``t`` for
          every nonzero ``s`` and ``|s| * K``. ``K`` is not normalised to ``K[2, 2] = 1``; divide by
          ``K[..., 2:, 2:]`` to normalise.
        - ``P`` must have exactly one batch dimension. float16 and bfloat16 raise.

    Args:
        P: the projection matrix with shape :math:`(B, 3, 4)`.
        eps: unused; kept for backward compatibility. The sign of the triangular factor's diagonal is taken
          exactly.

    Returns:
        - The Camera matrix with shape :math:`(B, 3, 3)`.
        - The Rotation matrix with shape :math:`(B, 3, 3)`.
        - The Translation vector with shape :math:`(B, 3, 1)`.

    """
    KORNIA_CHECK_SHAPE(P, ["*", "3", "4"])
    # P and -P are the same camera. Decompose the representative whose left block has positive determinant,
    # so that a positive-diagonal K leaves R a proper rotation.
    P = torch.where(torch.linalg.det(P[:, 0:3, 0:3])[:, None, None] < 0, -P, P)
    submat_3x3 = P[:, 0:3, 0:3]
    last_column = P[:, 0:3, 3].unsqueeze(-1)

    # Trick to turn QR-decomposition into RQ-decomposition
    reverse = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=P.device, dtype=P.dtype).unsqueeze(0)
    submat_3x3 = torch.matmul(reverse, submat_3x3).permute(0, 2, 1)
    ortho_mat, upper_mat = linalg_qr(submat_3x3)
    ortho_mat = torch.matmul(reverse, ortho_mat.permute(0, 2, 1))
    upper_mat = torch.matmul(reverse, torch.matmul(upper_mat.permute(0, 2, 1), reverse))

    # Turning the `upper_mat's` diagonal elements to positive (a zero entry keeps its sign: rank-deficient P).
    diagonals = torch.diagonal(upper_mat, dim1=-2, dim2=-1)
    signs = torch.where(diagonals < 0, -torch.ones_like(diagonals), torch.ones_like(diagonals))
    signs_mat = torch.diag_embed(signs)

    K = torch.matmul(upper_mat, signs_mat)
    R = torch.matmul(signs_mat, ortho_mat)
    t = torch.linalg.solve(K, last_column)

    return K, R, t


def depth_from_point(R: torch.Tensor, t: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    r"""Return the depth of a point transformed by a rigid transform.

    Convention:
        - Returns the ``z`` coordinate of ``R @ X + t``, the depth in the camera of :func:`projection_from_KRt`
          with the same ``R`` and ``t``. The sign is not checked. An ``X`` of shape ``(B, 3)`` with a batched
          ``R`` is read as ``B`` points seen by every camera and gives ``(B, B)``, not one point per camera.

    Args:
       R: The rotation matrix with shape :math:`(*, 3, 3)`.
       t: The translation vector with shape :math:`(*, 3, 1)`.
       X: The 3d points with shape :math:`(*, N, 3)`.

    Returns:
       The depth value per point with shape :math:`(*, N)`.

    """
    X_tmp = R @ X.transpose(-2, -1)
    return X_tmp[..., 2, :] + t[..., 2, :]


# adapted from:
# https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/fundamental.cpp#L61
# https://github.com/mapillary/OpenSfM/blob/master/opensfm/multiview.py#L14
def _nullspace(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the null space of A.

    Return the smallest singular value and the corresponding vector.
    """
    _, s, v = _torch_svd_cast(A)
    return s[..., -1], v[..., -1]


def projections_from_fundamental(F_mat: torch.Tensor) -> torch.Tensor:
    r"""Get the projection matrices from the Fundamental Matrix.

    Convention:
        - Returns the canonical pair for ``F`` in the ``x2^T F x1 = 0`` convention of
          :func:`~kornia.geometry.epipolar.find_fundamental`, stacked on the last dimension:
          ``[..., 0] = [I | 0]`` for the first image and ``[..., 1] = [[e2]_x F | e2]``, with ``e2`` the
          left null vector of ``F``, the epipole in the second image (``e2^T F = 0``).
        - ``F_mat`` must have exactly one batch dimension.

    Args:
       F_mat: the fundamental matrix with the shape :math:`(B, 3, 3)`.

    Returns:
        The projection matrices with shape :math:`(B, 3, 4, 2)`.

    """
    KORNIA_CHECK_SHAPE(F_mat, ["*", "3", "3"])

    R1 = eye_like(3, F_mat)  # Bx3x3
    t1 = vec_like(3, F_mat)  # Bx3

    Ft_mat = F_mat.transpose(-2, -1)

    _, e2 = _nullspace(Ft_mat)

    R2 = cross_product_matrix(e2) @ F_mat  # Bx3x3
    t2 = e2[..., :, None]  # Bx3x1

    P1 = torch.cat([R1, t1], dim=-1)  # Bx3x4
    P2 = torch.cat([R2, t2], dim=-1)  # Bx3x4

    return torch.stack([P1, P2], dim=-1)
