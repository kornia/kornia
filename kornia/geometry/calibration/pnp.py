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

"""Estimate a camera's world-to-camera pose from 3D-2D correspondences by the direct linear transform (PnP)."""

from typing import Optional, Tuple

import torch
from torch.linalg import qr as linalg_qr

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SAME_SHAPE, KORNIA_CHECK_SHAPE
from kornia.core.utils import (
    _at_least_float32,
    _autocast_disabled,
    _reduce_tall_cuda_svd,
    _right_singular_vector,
    _svd_work_dtype,
    _torch_linalg_svdvals,
    _torch_svd_right,
)
from kornia.geometry.conversions import convert_points_to_homogeneous, normalize_points_with_intrinsics


def _mean_isotropic_scale_normalize(
    points: torch.Tensor, eps: float = 1e-8, weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized points, their centroid and their isotropic scale."""
    KORNIA_CHECK_SHAPE(points, ["B", "N", "D"])
    # Center in a local frame before reducing, so the centroid's rounding at a
    # distant origin does not enter the fit or move an otherwise planar cloud.
    if weights is None:
        anchor = points[:, :1]
        local_points = points - anchor
        local_center = local_points.mean(dim=1, keepdim=True)
        centered = local_points - local_center
        mean_distance = centered.norm(dim=-1).mean(dim=1)
    else:
        anchor_idx = weights.argmax(dim=1)[:, None, None].expand(-1, 1, points.shape[-1])
        anchor = points.gather(dim=1, index=anchor_idx)
        local_points = points - anchor
        w = weights / weights.sum(dim=1, keepdim=True)
        local_center = (w[..., None] * local_points).sum(dim=1, keepdim=True)
        centered = local_points - local_center
        mean_distance = (w * centered.norm(dim=-1)).sum(dim=1)

    scale = points.shape[-1] ** 0.5 / (mean_distance + eps)
    return centered * scale[:, None, None], anchor + local_center, scale


def solve_pnp_dlt(
    world_points: torch.Tensor,
    img_points: torch.Tensor,
    intrinsics: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    svd_eps: float = 1e-4,
) -> torch.Tensor:
    r"""Estimate a camera's world-to-camera pose from 3D-2D correspondences by the DLT (PnP).

    ``Rt = solve_pnp_dlt(world_points, img_points, K)`` takes ``B`` sets of :math:`N \geq 6` world points
    ``(B, N, 3)``, their pixels ``(B, N, 2)`` and the intrinsics ``(B, 3, 3)``, and returns the world-to-camera
    extrinsics ``[R | t]`` of each set, ``(B, 3, 4)``: the Hartley-normalized, optionally weighted, linear
    (Direct Linear Transform) solution of the Perspective-n-Point problem, with ``R`` made orthonormal by a QR step.
    Orthonormalization is performed about the centroid of the world points so that re-expressing them in a
    translated world frame does not change their camera-space coordinates, up to rounding. It is not robust to
    outliers and does not refine the pose nonlinearly. For 2D-2D correspondences of a plane,
    use :func:`~kornia.geometry.homography.find_homography_dlt`; for two views with no known 3D points,
    :func:`~kornia.geometry.epipolar.find_essential`.

    This function cannot be used if all the 3D world points (of any element
    of the batch) lie on a line or if all the 3D world points (of any element
    of the batch) lie on a plane. This function attempts to check for these
    conditions and throws an AssertionError if found. Do note that this check
    is sensitive to the value of the svd_eps parameter.

    Another bad condition occurs when the camera and the points lie on a
    twisted cubic. However, this function does not check for this condition.

    Convention:
        - the returned ``(B, 3, 4)`` matrix is the **world-to-camera** ``[R | t]``: it maps a world point
          into the camera frame (``X_cam = X_world + (1, 0, 0)`` gives ``t = (+1, 0, 0)``), not a
          camera-to-world pose.
        - ``intrinsics`` is the ``(B, 3, 3)`` ``K``, or a ``(1, 3, 3)`` ``K`` shared by the batch; the
          ``(B, 4, 4)`` matrix that a :class:`~kornia.geometry.camera.pinhole.PinholeCamera` stores is rejected.
        - ``weights`` scales the two rows each point contributes to the homogeneous linear system, and the
          normalization and the degeneracy check use the same weighting. A uniform ``weights`` leaves the answer
          unchanged up to rounding, and a zero weight is the same as removing the point.
        - fewer than 6 points or 6 nonzero ``weights``, or a ``world_points``, ``img_points`` or ``intrinsics``
          in a dtype other than float16/bfloat16/float32/float64, raise :class:`~kornia.core.exceptions.BaseError`
          naming the argument; a degenerate ``world_points`` raises :class:`AssertionError` from the check above.
        - float16 and bfloat16 inputs are solved in float32 and the pose is returned in the dtype of
          ``world_points``.
        - a caller's autocast region does not change the result: float32 inputs are solved in float32.

    Args:
        world_points : A torch.Tensor with shape ``(B, N, 3)`` representing
          the points in the world space.
        img_points : A torch.Tensor with shape ``(B, N, 2)`` representing
          the points in the image space.
        intrinsics : A torch.Tensor with shape ``(B, 3, 3)`` representing
          the intrinsic matrices.
        weights : A torch.Tensor with shape ``(B, N)`` representing the
            weights for each point. If None, all points are considered to be equally important.
        svd_eps : A small float value to avoid numerical precision issues.

    Returns:
        A torch.Tensor with shape ``(B, 3, 4)`` representing the estimated world to
        camera transformation matrices (also known as the extrinsic matrices).

    See Also:
        - :func:`~kornia.geometry.homography.find_homography_dlt`: the DLT for 2D-2D correspondences, a homography
          rather than a pose.
        - :func:`~kornia.geometry.epipolar.find_essential`: the relative pose of two views from 2D-2D
          correspondences, without 3D points.
        - :func:`~kornia.geometry.epipolar.projection_from_KRt`: the ``(B, 3, 4)`` projection matrix
          ``K [R | t]`` of the result.
        - :class:`~kornia.geometry.camera.pinhole.PinholeCamera`: a camera model that takes the result, padded to
          ``(B, 4, 4)``, as its extrinsics.

    Example:
        >>> world_points = torch.tensor([[
        ...     [ 5. , -5. ,  0. ], [ 0. ,  0. ,  1.5],
        ...     [ 2.5,  3. ,  6. ], [ 9. , -2. ,  3. ],
        ...     [-4. ,  5. ,  2. ], [-5. ,  5. ,  1. ],
        ... ]], dtype=torch.float64)
        >>>
        >>> img_points = torch.tensor([[
        ...     [1409.1504, -800.936 ], [ 407.0207, -182.1229],
        ...     [ 392.7021,  177.9428], [1016.838 ,   -2.9416],
        ...     [ -63.1116,  142.9204], [-219.3874,   99.666 ],
        ... ]], dtype=torch.float64)
        >>>
        >>> intrinsics = torch.tensor([[
        ...     [ 500.,    0.,  250.],
        ...     [   0.,  500.,  250.],
        ...     [   0.,    0.,    1.],
        ... ]], dtype=torch.float64)
        >>>
        >>> print(world_points.shape, img_points.shape, intrinsics.shape)
        torch.Size([1, 6, 3]) torch.Size([1, 6, 2]) torch.Size([1, 3, 3])
        >>>
        >>> pred_world_to_cam = kornia.geometry.solve_pnp_dlt(world_points, img_points, intrinsics)
        >>>
        >>> print(pred_world_to_cam.shape)
        torch.Size([1, 3, 4])
        >>>
        >>> pred_world_to_cam
        tensor([[[ 0.9392, -0.3432, -0.0130,  1.6734],
                 [ 0.3390,  0.9324, -0.1254, -4.3634],
                 [ 0.0552,  0.1134,  0.9920,  3.7785]]], dtype=torch.float64)

    """
    # This function was implemented based on ideas inspired from multiple references.
    # ============
    # References:
    # ============
    # 1. https://team.inria.fr/lagadic/camera_localization/tutorial-pose-dlt-opencv.html
    # 2. https://github.com/opencv/opencv/blob/68d15fc62edad980f1ffa15ee478438335f39cc3/modules/calib3d/src/calibration.cpp
    # 3. http://rpg.ifi.uzh.ch/docs/teaching/2020/03_camera_calibration.pdf
    # 4. http://www.cs.cmu.edu/~16385/s17/Slides/11.3_Pose_Estimation.pdf
    # 5. https://www.ece.mcmaster.ca/~shirani/vision/hartley_ch7.pdf
    accepted_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    accepted = "float16, bfloat16, float32 or float64"
    KORNIA_CHECK_IS_TENSOR(world_points)
    KORNIA_CHECK_IS_TENSOR(img_points)
    KORNIA_CHECK_IS_TENSOR(intrinsics)
    KORNIA_CHECK(isinstance(svd_eps, float), f"svd_eps must be a float, got {type(svd_eps).__name__}.")
    KORNIA_CHECK(world_points.dtype in accepted_dtypes, f"world_points must be {accepted}, got {world_points.dtype}.")
    KORNIA_CHECK(img_points.dtype in accepted_dtypes, f"img_points must be {accepted}, got {img_points.dtype}.")
    KORNIA_CHECK(intrinsics.dtype in accepted_dtypes, f"intrinsics must be {accepted}, got {intrinsics.dtype}.")
    # No backend factorizes half precision: half-precision inputs are solved in float32 and the pose is rounded once.
    dtype = world_points.dtype
    world_points, img_points = _at_least_float32(world_points), _at_least_float32(img_points)
    intrinsics = _at_least_float32(intrinsics)
    KORNIA_CHECK_SHAPE(world_points, ["B", "N", "3"])
    KORNIA_CHECK_SHAPE(img_points, ["B", "N", "2"])
    KORNIA_CHECK_SHAPE(intrinsics, ["B", "3", "3"])
    KORNIA_CHECK_SAME_SHAPE(world_points[:, :, 0], img_points[:, :, 0])
    KORNIA_CHECK(
        world_points.shape[1] >= 6,
        lambda: f"world_points must hold at least 6 points (N >= 6), got N = {world_points.shape[1]}.",
    )
    if weights is not None:
        KORNIA_CHECK_IS_TENSOR(weights)
        weights = _at_least_float32(weights)

    B, N = world_points.shape[:2]

    # The weights scale each point's rows of the linear system, so the normalization and the
    # degeneracy check below use the same weighting: a point with weight zero drops out of all three.
    point_weights = None
    check_scale = None
    if weights is not None:
        if weights.shape != (B, N):
            raise AssertionError(f"Weights should have shape (B, N). Got {weights.shape}.")
        num_used = (weights != 0).sum(dim=1, keepdim=True)
        KORNIA_CHECK(
            bool((num_used >= 6).all()),
            "weights must hold at least 6 nonzero entries in every batch element, as world_points must hold 6 points.",
        )
        # Only the weights relative to the largest one matter below; dividing by it first keeps the squares
        # from overflowing or underflowing.
        rel_weights = weights / weights.abs().amax(dim=1, keepdim=True)
        point_weights = rel_weights**2
        # Rescale so that the squared weights sum to the number of nonzero-weight points: uniform weights
        # leave the check unchanged, and zero weights check the same as the remaining points alone.
        num_used = num_used.to(point_weights.dtype)
        check_scale = (rel_weights * torch.sqrt(num_used / point_weights.sum(dim=1, keepdim=True)))[..., None]

    # A caller's autocast region would run the products below in half precision: torch.det, which has no half-precision
    # CPU kernel, then raised, and float32 inputs would no longer be solved in float32.
    with _autocast_disabled(world_points.device.type):
        # Getting normalized world points.
        world_points_norm, world_center, world_scale = _mean_isotropic_scale_normalize(
            world_points, weights=point_weights
        )

        # Checking if world_points_norm (of any element of the batch) has rank = 3. This
        # function cannot be used if all world points (of any element of the batch) lie
        # on a line or if all world points (of any element of the batch) lie on a plane.
        rank_system = world_points_norm if check_scale is None else check_scale * world_points_norm
        if _reduce_tall_cuda_svd(rank_system):
            # Only the rank decision uses these values, so no QR backward is needed.
            # Keep the original svdvals work dtype and the unchanged CPU/MPS paths.
            _, rank_system = torch.linalg.qr(rank_system.detach(), mode="r")
        s = _torch_linalg_svdvals(rank_system)
        if torch.any(s[:, -1] < svd_eps):
            raise AssertionError(
                "The last singular value of one/more of the elements of the batch is smaller "
                f"than {svd_eps}. This function cannot be used if all world_points (of any "
                "element of the batch) lie on a line or if all world_points (of any "
                "element of the batch) lie on a plane."
            )
        world_points_norm_h = convert_points_to_homogeneous(world_points_norm)

        # Normalizing img_points_inv
        img_points_inv = normalize_points_with_intrinsics(img_points, intrinsics)
        img_points_norm, img_center, img_scale = _mean_isotropic_scale_normalize(img_points_inv, weights=point_weights)

        # Setting up the system (the matrix A in Ax=0)
        system = torch.zeros((B, 2 * N, 12), dtype=world_points.dtype, device=world_points.device)
        system[:, 0::2, 0:4] = world_points_norm_h
        system[:, 1::2, 4:8] = world_points_norm_h
        system[:, 0::2, 8:12] = world_points_norm_h * (-1) * img_points_norm[..., 0:1]
        system[:, 1::2, 8:12] = world_points_norm_h * (-1) * img_points_norm[..., 1:2]

        # Apply weights to the system if provided
        if weights is not None:
            weights_expanded = weights.unsqueeze(-1).repeat(1, 1, 2).view(B, 2 * N, 1)
            # Multiply the system matrix by the expanded weights
            system = weights_expanded * system

        # Getting the solution vectors.
        # Only the right singular vectors are used; reduced SVD avoids saving the unused (2N, 2N) U.
        # The SVD is of the detached system: the solution's gradient comes from _right_singular_vector, which stays
        # finite when other singular values are equal, as for world points on the corners of a cube.
        work = system.detach().to(_svd_work_dtype(system))
        s, v = _torch_svd_right(work)
        solution = _right_singular_vector(system, s, v, -1).to(system.dtype)

        # Undo image normalization. Keep the 3D origin at the centroid until
        # the rotation is orthonormal: applying the world shift with the raw
        # DLT matrix would amplify its non-rigid component as the origin moves.
        solution = solution.reshape(B, 3, 4)
        homogeneous_row = solution[:, 2:3]
        solution = torch.cat(
            (solution[:, :2] / img_scale[:, None, None] + img_center.mT * homogeneous_row, homogeneous_row), dim=1
        )
        linear = solution[:, :, :3] * world_scale[:, None, None]

        # Choose the sign and scale of the homogeneous solution using the rotation.
        det = torch.det(linear)
        sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
        factor = (sign / linear[:, :, 0].norm(dim=1))[:, None, None]

        rotation, triangular = linalg_qr(linear * factor)
        rotation = rotation * triangular.diagonal(dim1=-2, dim2=-1).sign()[:, None, :]
        translation = solution[:, :, 3:] * factor - rotation @ world_center.mT
        return torch.cat((rotation, translation), dim=-1).to(dtype)
