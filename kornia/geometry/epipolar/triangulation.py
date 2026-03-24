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

"""Module with the functionalities for triangulation."""

from __future__ import annotations

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.utils import _normalize_to_float32_or_float64, _torch_svd_cast, is_mps_tensor_safe
from kornia.geometry.conversions import convert_points_from_homogeneous
from kornia.geometry.solvers import null_vector_3x4

# https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/triangulation.cpp#L68

# cuSOLVER's batched symmetric eigenvalue solver crashes above this many 4x4 matrices
# in a single call (empirically observed at >=32 K on current CUDA/PyTorch versions).
_CUSOLVER_EIGH_BATCH_LIMIT: int = 28_000


def _eigh_smallest_vec(M: torch.Tensor) -> torch.Tensor:
    """Return the eigenvector for the smallest eigenvalue of each symmetric matrix.

    Handles cuSOLVER's batch-size limit by chunking when necessary.

    Args:
        M: batch of symmetric PSD matrices, shape ``(N, k, k)``.

    Returns:
        Eigenvectors of shape ``(N, k)``.
    """
    N = M.shape[0]
    if N <= _CUSOLVER_EIGH_BATCH_LIMIT:
        _, V = torch.linalg.eigh(M)
        return V[..., 0]

    parts = [
        torch.linalg.eigh(M[i : i + _CUSOLVER_EIGH_BATCH_LIMIT])[1][..., 0]
        for i in range(0, N, _CUSOLVER_EIGH_BATCH_LIMIT)
    ]
    return torch.cat(parts, dim=0)


def triangulate_points(
    P1: torch.Tensor,
    P2: torch.Tensor,
    points1: torch.Tensor,
    points2: torch.Tensor,
    solver: str = "eigh",
) -> torch.Tensor:
    r"""Reconstructs a bunch of points by triangulation.

    Triangulates the 3d position of 2d correspondences between several images.
    Reference: Internally it uses DLT formulation from Hartley/Zisserman 12.2 pag.312

    The input points are assumed to be in homogeneous coordinate system and being inliers
    correspondences. The method does not perform any robust estimation.

    Args:
        P1: The projection matrix for the first camera with shape :math:`(*, 3, 4)`.
        P2: The projection matrix for the second camera with shape :math:`(*, 3, 4)`.
        points1: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        points2: The set of points seen from the second camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        solver: Back-end used to find the null vector of the :math:`4 \times 4` DLT
          constraint matrix. One of:

          * ``"svd"`` — most numerically stable. Promotes to fp64 and uses a full
            SVD (via :func:`~kornia.core.utils._torch_svd_cast`). Suitable when
            maximum accuracy is required regardless of speed.
          * ``"eigh"`` *(default)* — forms :math:`X^\top X` and finds the eigenvector
            for its smallest eigenvalue via :func:`torch.linalg.eigh`. Algebraically
            equivalent to the SVD solution; slightly less numerically stable because
            forming :math:`X^\top X` squares the singular values. Typically **10-26x
            faster** than ``"svd"`` on GPU for large batches.
          * ``"cofactor"`` — solves two :math:`3 \times 4` sub-systems analytically
            using :func:`~kornia.geometry.solvers.null_vector_3x4` (closed-form
            cofactor expansion, no LAPACK call). The two solutions are averaged after
            normalisation. This matches the full DLT solution when the constraint
            system is exactly consistent, but is only an approximation in the noisy
            inconsistent case. Fastest option for all batch sizes.

    Returns:
        The reconstructed 3d points in the world frame with shape :math:`(*, N, 3)`.

    Example:
        >>> P1 = torch.eye(3, 4)[None]   # 1x3x4
        >>> P2 = torch.eye(3, 4)[None]
        >>> pts1 = torch.rand(1, 5, 2)
        >>> pts2 = torch.rand(1, 5, 2)
        >>> pts3d = triangulate_points(P1, P2, pts1, pts2)
        >>> pts3d.shape
        torch.Size([1, 5, 3])

    """
    KORNIA_CHECK_SHAPE(P1, ["*", "3", "4"])
    KORNIA_CHECK_SHAPE(P2, ["*", "3", "4"])
    KORNIA_CHECK_SHAPE(points1, ["*", "N", "2"])
    KORNIA_CHECK_SHAPE(points2, ["*", "N", "2"])

    # Build the four DLT constraint rows (each (*, N, 4)) via vectorized broadcasting.
    # P[..., r:r+1, :] broadcasts with points[..., c:c+1] → (*, N, 4).
    row0 = points1[..., 0:1] * P1[..., 2:3, :] - P1[..., 0:1, :]  # (*, N1, 4)
    row1 = points1[..., 1:2] * P1[..., 2:3, :] - P1[..., 1:2, :]  # (*, N1, 4)
    row2 = points2[..., 0:1] * P2[..., 2:3, :] - P2[..., 0:1, :]  # (*, N2, 4)
    row3 = points2[..., 1:2] * P2[..., 2:3, :] - P2[..., 1:2, :]  # (*, N2, 4)
    # Unify N1 and N2: one may be 1 when points1/points2 are broadcast-compatible.
    row0, row1, row2, row3 = torch.broadcast_tensors(row0, row1, row2, row3)

    if solver == "svd":
        X = torch.stack([row0, row1, row2, row3], dim=-2)  # (*, N, 4, 4)
        # SVD: last right singular vector minimises ||Ax|| s.t. ||x||=1.
        # _torch_svd_cast promotes to fp64 for numerical stability and returns V
        # with singular vectors as columns; the last column corresponds to the
        # smallest singular value.
        _, _, V = _torch_svd_cast(X)
        points3d_h = V[..., -1]  # (*, N, 4)

    elif solver == "eigh":
        X = torch.stack([row0, row1, row2, row3], dim=-2)  # (*, N, 4, 4)
        # Solve the homogeneous least-squares problem min ||Ax|| s.t. ||x||=1.
        # The minimiser is the eigenvector of X^T X associated with its smallest
        # eigenvalue.  This is algebraically equivalent to the last right singular
        # vector of X used in SVD-based DLT, though forming X^T X can be less
        # numerically stable than a direct SVD.  The result is defined up to sign,
        # which is fine for homogeneous coordinates.
        # The approach is valid in both the noise-free (rank-3) and the noisy
        # inconsistent case, where the rows do not share an exact nullspace.
        # Mirror _torch_svd_cast's promotion rules so numerical behaviour is
        # comparable to the "svd" solver: fp32 → fp64 for stability, fp16/bf16 →
        # fp32, fp64 stays, MPS capped at fp32 (no fp64 support there).
        if is_mps_tensor_safe(X):
            compute_dtype = torch.float32
        elif X.dtype == torch.float32:
            compute_dtype = torch.float64
        else:
            compute_dtype = _normalize_to_float32_or_float64(X.dtype)
        batch_shape = X.shape[:-2]  # (*, N)
        X_cast = X.to(compute_dtype)
        XTX = X_cast.mT @ X_cast  # (*, N, 4, 4) symmetric PSD
        flat = XTX.flatten(0, -3)  # (M, 4, 4)
        v_flat = _eigh_smallest_vec(flat).to(X.dtype)  # (M, 4)
        points3d_h = v_flat.reshape(*batch_shape, 4)  # (*, N, 4)

    elif solver == "cofactor":
        # Solve two 3x4 sub-systems analytically via cofactor expansion and
        # average the normalised results.  This matches the full DLT solution
        # when the constraint system is exactly consistent (noise-free), but is
        # only an approximation in the noisy inconsistent case.
        # null_vector_3x4 uses only arithmetic ops, so promote fp16/bf16 → fp32.
        compute_dtype = _normalize_to_float32_or_float64(row0.dtype)
        r0 = row0.to(compute_dtype)
        r1 = row1.to(compute_dtype)
        r2 = row2.to(compute_dtype)
        r3 = row3.to(compute_dtype)
        A_012 = torch.stack([r0, r1, r2], dim=-2)  # (*, N, 3, 4)
        A_013 = torch.stack([r0, r1, r3], dim=-2)  # (*, N, 3, 4)
        h_012 = null_vector_3x4(A_012).to(row0.dtype)  # (*, N, 4)
        h_013 = null_vector_3x4(A_013).to(row0.dtype)  # (*, N, 4)
        n012 = h_012.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        n013 = h_013.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v012 = h_012 / n012
        v013 = h_013 / n013
        # Null vectors are defined up to a global sign; align signs before averaging
        # to prevent cancellation when the two sub-system solutions point in opposite
        # directions (which would yield a near-zero homogeneous vector and NaN after
        # conversion from homogeneous coordinates).
        dot = (v012 * v013).sum(dim=-1, keepdim=True)
        v013 = torch.where(dot < 0, -v013, v013)
        points3d_h = v012 + v013  # (*, N, 4)

    else:
        raise NotImplementedError(f"Unknown solver '{solver}'. Choose from: 'svd', 'eigh', 'cofactor'.")

    points3d: torch.Tensor = convert_points_from_homogeneous(points3d_h)
    return points3d
