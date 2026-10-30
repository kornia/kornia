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

"""Camera intrinsics and projection matrices: compose, decompose and build them for two-view geometry."""

from typing import Tuple, Union

import torch
from torch.linalg import qr as linalg_qr

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.ops import eye_like, vec_like
from kornia.core.utils import _autocast_disabled, _torch_svd_cast

from .numeric import cross_product_matrix


def intrinsics_like(focal: float, input: torch.Tensor) -> torch.Tensor:
    r"""Build a pinhole camera matrix per image of a batch from one focal length and the image size.

    ``K = intrinsics_like(focal, input)`` takes a focal length in pixels and a floating-point image batch
    ``(B, C, H, W)`` and returns ``(B, 3, 3)`` intrinsics with ``fx = fy = focal``, zero skew and the principal point
    ``(W / 2, H / 2)``, in the dtype and on the device of the images. That principal point is the half-pixel centre,
    not the integer-pixel centre of kornia's grids (see the known defect below). For random intrinsics, use
    :func:`random_intrinsics`; to resize existing ones with their image, :func:`scale_intrinsics`.

    Convention:
        - ``focal`` must be positive and the input 4-D.
        - The input must be floating point, like kornia's images: an integer input raises by design rather than
          returning an integer ``K``.
        - Known defects: ``(W / 2, H / 2)`` is the half-pixel centre, not the integer-pixel centre
          ``((W - 1) / 2, (H - 1) / 2)`` of kornia's grids
          (`#4263 <https://github.com/kornia/kornia/issues/4263>`_).

    Args:
        focal: the focal length for the camera matrix.
        input: image tensor that will determine the batch size and image height
          and width. It is assumed to be a floating-point tensor in the shape of ``(B, C, H, W)``.

    Returns:
        The camera matrix with the shape of ``(B, 3, 3)``.

    See Also:
        - :func:`random_intrinsics`: random focal lengths and principal point instead.
        - :func:`scale_intrinsics`: the intrinsics of a resized image.
        - :class:`~kornia.geometry.camera.pinhole.PinholeCamera`: a camera model that holds intrinsics and
          extrinsics together.

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
    r"""Draw a random pinhole camera matrix with focal lengths and principal point uniform in one range.

    ``K = random_intrinsics(low, high)`` returns a ``(1, 3, 3)`` camera matrix with zero skew whose ``fx``, ``fy``,
    ``cx`` and ``cy`` are four independent draws from :math:`U(low, high)`, for synthetic data such as that of
    :func:`generate_scene`. For deterministic intrinsics from a focal length and an image size, use
    :func:`intrinsics_like`.

    Convention:
        - The draws come from torch's global generator. The bounds are scalars; dtype and device follow tensor
          bounds, and Python floats give the default dtype and device.

    Args:
        low: lower range (inclusive).
        high: upper range (exclusive).

    Returns:
        the random camera matrix with the shape of ``(1, 3, 3)``.

    See Also:
        - :func:`intrinsics_like`: deterministic intrinsics from a focal length and an image size.
        - :func:`generate_scene`: a random scene whose cameras share one such matrix.

    """
    sampler = torch.distributions.Uniform(low, high)
    params = sampler.sample((4,))
    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    camera_matrix = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=fx.dtype, device=fx.device)
    return camera_matrix.unsqueeze(0)


def scale_intrinsics(camera_matrix: torch.Tensor, scale_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Scale the focal lengths and principal point of camera matrices by an image resize factor.

    ``K_scaled = scale_intrinsics(camera_matrix, scale_factor)`` takes camera matrices ``(B, 3, 3)`` and a factor, a
    float or a ``(B,)`` tensor, and returns a new tensor of the input's shape with ``fx``, ``fy``, ``cx`` and ``cy``
    multiplied by it, for an image resized by that factor. This is the rule of
    :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.scale`, with its known defects (below): the principal point
    does not follow kornia's integer pixel centres, and the skew is left unscaled. For the camera of a whole
    :class:`~kornia.geometry.camera.pinhole.PinholeCamera`, use that method.

    Convention:
        - Applies the rule of :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.scale` and returns a new tensor.
        - Known defects: the principal-point rule of
          :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.scale`, and the skew ``K[0, 1]``, which is left
          unscaled although resizing by ``scale_factor`` scales it
          (`#4263 <https://github.com/kornia/kornia/issues/4263>`_).

    Args:
        camera_matrix: the camera calibration matrix containing the intrinsic
          parameters. The expected shape for the tensor is ``(B, 3, 3)``.
        scale_factor: the scaling factor to be applied: a float, or a tensor of shape ``(B,)`` that
          scales each batch element by its own factor.

    Returns:
        The scaled camera matrix with the same shape as the input ``(B, 3, 3)``.

    See Also:
        - :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.scale`: the same rule on a camera model.
        - :func:`intrinsics_like`: new intrinsics from a focal length and an image size.

    """
    K_scale = camera_matrix.clone()
    K_scale[..., 0, 0] *= scale_factor
    K_scale[..., 1, 1] *= scale_factor
    K_scale[..., 0, 2] *= scale_factor
    K_scale[..., 1, 2] *= scale_factor
    return K_scale


def projection_from_KRt(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""Compose the 3x4 projection matrix P = K [R | t] of intrinsics and a world-to-camera pose.

    ``P = projection_from_KRt(K, R, t)`` takes intrinsics ``(*, 3, 3)``, world-to-camera rotations ``(*, 3, 3)``
    and translations ``(*, 3, 1)`` and returns the projection matrices :math:`P = K [R|t]`, ``(*, 3, 4)``, that take
    homogeneous world points to homogeneous pixels. For the reverse, use :func:`KRt_from_projection`; for
    ``[R | t]`` without intrinsics, :func:`~kornia.geometry.transform.projection_from_Rt`.

    Convention:
        - ``R`` and ``t`` take world points into the camera frame, ``X_cam = R @ X + t``, and ``K`` takes the
          camera frame to pixels. ``t`` is the extrinsic translation; the camera centre is ``-R.T @ t``.
        - ``K``, ``R`` and ``t`` must have the same number of dimensions, and ``R`` and ``t`` the same batch shape;
          ``K`` broadcasts against them. :func:`depth_from_point` gives the depth of a point in this camera.

    Args:
       K: the camera matrix with the intrinsics with shape ``(*, 3, 3)``.
       R: The rotation matrix with shape ``(*, 3, 3)``.
       t: The translation vector with shape ``(*, 3, 1)``.

    Returns:
       The projection matrix P with shape ``(*, 3, 4)``.

    See Also:
        - :func:`KRt_from_projection`: the reverse, ``K``, ``R`` and ``t`` of a projection matrix.
        - :func:`~kornia.geometry.transform.projection_from_Rt`: ``[R | t]`` alone, without intrinsics.
        - :func:`depth_from_point`: the depth of points in the camera of this matrix.
        - :func:`triangulate_points`: 3D points from two such matrices and matched pixels.

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
    r"""Decompose a 3x4 projection matrix into intrinsics, rotation and translation, with K [R | t] = +-P.

    ``K, R, t = KRt_from_projection(P)`` takes projection matrices ``(B, 3, 4)`` and returns, by an RQ decomposition
    of their left ``3 x 3`` block, the intrinsics ``K`` ``(B, 3, 3)``, upper triangular with a positive diagonal, the
    world-to-camera rotation ``R`` ``(B, 3, 3)`` with ``det R = 1`` and the translation ``t`` ``(B, 3, 1)``, not the
    camera centre: the inverse of :func:`projection_from_KRt` up to the scale and sign of ``P``. ``K`` keeps the scale
    of ``P``; divide it by ``K[..., 2:, 2:]`` for ``K[2, 2] = 1``.

    Convention:
        - :ref:`Two-view geometry <two-view-conventions>` compares this with OpenCV's
          ``decomposeProjectionMatrix``, which returns the camera centre.
        - ``P`` is defined up to a nonzero scale, sign included: ``s * P`` gives the same ``R`` and ``t`` for
          every nonzero ``s`` and ``|s| * K``. So :math:`K [R|t]` is ``P`` when ``P[:, :3, :3]`` has positive
          determinant and ``-P`` when it has negative determinant (``-P`` gives the factors of ``P``). ``K`` is
          not normalised to ``K[2, 2] = 1``; divide by ``K[..., 2:, 2:]`` to normalise.
        - ``P`` must have exactly one batch dimension. float16 and bfloat16 raise.
        - A caller's autocast region does not change the result, outside TorchScript.

    Args:
        P: the projection matrix with shape ``(B, 3, 4)``.
        eps: unused; kept for backward compatibility. The sign of the triangular factor's diagonal is taken
          exactly.

    Returns:
        - The Camera matrix with shape ``(B, 3, 3)``.
        - The Rotation matrix with shape ``(B, 3, 3)``.
        - The Translation vector with shape ``(B, 3, 1)``.

    See Also:
        - :func:`projection_from_KRt`: the reverse, ``P = K [R | t]``.
        - :func:`projections_from_fundamental`: a projective camera pair consistent with a fundamental matrix; the
          left ``3 x 3`` block of its second camera has rank 2, so this decomposition gives no metric camera for it.

    """
    KORNIA_CHECK_SHAPE(P, ["*", "3", "4"])
    # A caller's autocast region would run the products below in half precision, which torch.linalg.det (CPU) and
    # qr (MPS) have no kernel for. TorchScript has no context managers; a scripted call keeps the caller's precision.
    if torch.jit.is_scripting():
        return _KRt_from_projection(P)
    with _autocast_disabled(P.device.type):
        return _KRt_from_projection(P)


def _KRt_from_projection(P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # P and -P are the same camera. With a positive-diagonal K, a left block of negative determinant leaves
    # det R = -1; negating R and t decomposes -P instead, so R is a rotation. The sign comes from the orthogonal
    # factor, whose determinant is +-1 at any scale of P, not from det(P[:, :3, :3]), which scales as s**3 and
    # underflows to -0.0 for a small negative s.
    flip = torch.linalg.det(R)[:, None, None] < 0
    R = torch.where(flip, -R, R)
    t = torch.where(flip, -t, t)

    return K, R, t


def depth_from_point(R: torch.Tensor, t: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    r"""Compute the camera-frame depth of 3D points under a world-to-camera rotation and translation.

    ``depth = depth_from_point(R, t, X)`` takes a world-to-camera rotation ``(*, 3, 3)`` and translation
    ``(*, 3, 1)`` and points ``(*, N, 3)`` and returns the ``z`` coordinate of ``R @ X + t``, ``(*, N)``: the depth
    of each point in the camera of :func:`projection_from_KRt` with the same ``R`` and ``t``, positive in front of
    it. :func:`motion_from_essential_choose_solution` tests its candidate poses with it.

    Convention:
        - The sign is not checked. An ``X`` of shape ``(B, 3)`` with a batched ``R`` is read as ``B`` points seen
          by every camera and gives ``(B, B)``, not one point per camera.

    Args:
       R: The rotation matrix with shape ``(*, 3, 3)``.
       t: The translation vector with shape ``(*, 3, 1)``.
       X: The 3d points with shape ``(*, N, 3)``.

    Returns:
       The depth value per point with shape ``(*, N)``.

    See Also:
        - :func:`projection_from_KRt`: the projection matrix of the camera this depth is measured in.
        - :func:`triangulate_points`: the 3D points of matched pixels, whose depths this measures.

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
    r"""Build a canonical pair of projection matrices consistent with a fundamental matrix.

    ``P = projections_from_fundamental(F_mat)`` takes fundamental matrices ``(B, 3, 3)`` in the
    :math:`x_2^\top F x_1 = 0` order of :func:`find_fundamental` and returns the canonical camera pair of Hartley
    and Zisserman, ``(B, 3, 4, 2)`` with the two matrices stacked on the last dimension, for which
    :func:`fundamental_from_projections` gives back ``F`` up to scale. The pair is determined only up to a
    projective transformation of the world, so it is not a metric pose; for the rotation and translation of
    calibrated cameras, use :func:`motion_from_essential_choose_solution` on an essential matrix.

    Convention:
        - ``[..., 0] = [I | 0]`` for the first image and ``[..., 1] = [[e2]_x F | e2]``, with ``e2`` the
          left null vector of ``F``, the epipole in the second image (``e2^T F = 0``).
        - ``F_mat`` must have exactly one batch dimension.

    Args:
       F_mat: the fundamental matrix with the shape ``(B, 3, 3)``.

    Returns:
        The projection matrices with shape ``(B, 3, 4, 2)``.

    See Also:
        - :func:`fundamental_from_projections`: the reverse, the fundamental matrix of two projection matrices.
        - :func:`triangulate_points`: a projective reconstruction of matched pixels from this pair.
        - :func:`KRt_from_projection`: the decomposition of a metric projection matrix.

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
