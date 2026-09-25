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

    Triangulates the 3d position of 2d correspondences between two images.
    Reference: the ``"svd"`` and ``"eigh"`` solvers use the DLT formulation from Hartley/Zisserman 12.2 pag.312.

    The input points are assumed to be inlier correspondences. The method does not perform any robust
    estimation.

    Convention:
        - ``P1`` pairs with ``points1`` and ``P2`` with ``points2``. Input and output points are Euclidean;
          :ref:`Two-view geometry <two-view-conventions>` maps this onto OpenCV. The leading
          dimensions of ``P1`` and ``P2`` broadcast against those of the points.
        - Cheirality and baseline are not checked: a point behind a camera is returned with negative depth, and with
          zero baseline the depth is undefined: ``"svd"`` and ``"eigh"`` return an arbitrary point on the line of
          sight, possibly behind the camera or at infinity (NaN), and ``"cofactor"`` returns NaN.
        - A correspondence at infinity, whose homogeneous ``w`` is at the roundoff of the input and compute dtypes,
          returns NaN. In float16 and bfloat16 that roundoff also covers a point a few tens of units away in the
          scale of ``t``.
        - ``"svd"`` and ``"eigh"`` solve in float64, or in float32 for float16 and bfloat16 input and on MPS;
          ``"cofactor"`` solves in float32, or in float64 for float64 input. All return the input dtype;
          ``solver`` below compares their accuracy.

    Args:
        P1: The projection matrix for the first camera with shape :math:`(*, 3, 4)`.
        P2: The projection matrix for the second camera with shape :math:`(*, 3, 4)`.
        points1: The set of points seen from the first camera, in the image coordinates of ``P1`` (pixels
          for ``P1 = K [R | t]``), with shape :math:`(*, N, 2)`.
        points2: The set of points seen from the second camera, in the image coordinates of ``P2`` (pixels
          for ``P2 = K [R | t]``), with shape :math:`(*, N, 2)`.
        solver: Back-end used to find the null vector of the :math:`4 \times 4` DLT
          constraint matrix. One of:

          * ``"svd"`` — most numerically stable. Uses a full SVD. Suitable when
            maximum accuracy is required regardless of speed.
          * ``"eigh"`` *(default)* — forms :math:`X^\top X` and finds the eigenvector
            for its smallest eigenvalue via :func:`torch.linalg.eigh`. Algebraically
            equivalent to the SVD solution, and equal to it to roundoff on well-conditioned
            input; forming :math:`X^\top X` squares the singular values, so on ill-conditioned
            rows, such as a baseline much shorter than the depth, it loses accuracy that
            ``"svd"`` keeps. Typically **10-26x
            faster** than ``"svd"`` on GPU for large batches.
          * ``"cofactor"`` — solves two :math:`3 \times 4` sub-systems analytically
            using :func:`~kornia.geometry.solvers.null_vector_3x4` (closed-form
            cofactor expansion, no LAPACK call). The two solutions are averaged after
            normalisation; a rank-deficient sub-system (zero baseline, or a pure ``x``
            translation with ``R = I`` and one ``K``, which makes two DLT rows coincide)
            is left out, and the point is NaN when both are. This matches the full DLT
            solution when the constraint system is exactly consistent, but is only an
            approximation in the noisy inconsistent case. Fastest option for all batch
            sizes.

    Returns:
        The reconstructed 3d points in the world frame with shape :math:`(*, N, 3)`.

    Example:
        >>> P1 = torch.eye(3, 4)[None]   # 1x3x4
        >>> P2 = torch.eye(3, 4)[None]
        >>> P2[..., 0, 3] = -1.0  # second camera shifted along x
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

    # svd and eigh mirror _torch_svd_cast's promotion: fp32 -> fp64 for stability, fp16/bf16 -> fp32, fp64
    # stays, MPS capped at fp32 (no fp64 support there). cofactor uses arithmetic only and keeps fp32.
    if is_mps_tensor_safe(row0):
        compute_dtype = torch.float32
    elif row0.dtype == torch.float32:
        compute_dtype = torch.float64
    else:
        compute_dtype = _normalize_to_float32_or_float64(row0.dtype)

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
        batch_shape = X.shape[:-2]  # (*, N)
        X_cast = X.to(compute_dtype)
        XTX = X_cast.mT @ X_cast  # (*, N, 4, 4) symmetric PSD
        flat = XTX.flatten(0, -3)  # (M, 4, 4)
        v_flat = _eigh_smallest_vec(flat).to(X.dtype)  # (M, 4)
        points3d_h = v_flat.reshape(*batch_shape, 4)  # (*, N, 4)

    elif solver == "cofactor":
        # Solve two 3x4 sub-systems analytically via cofactor expansion and
        # average the sign-aligned normalised results.  This matches the full
        # DLT solution when the constraint system is exactly consistent
        # (noise-free), but is only an approximation in the noisy case.
        # null_vector_3x4 uses only arithmetic ops, so promote fp16/bf16 -> fp32 and stay there until the
        # null vectors are normalised: the unnormalised cofactors of pixel-scale rows overflow float16.
        compute_dtype = _normalize_to_float32_or_float64(row0.dtype)
        r0 = row0.to(compute_dtype)
        r1 = row1.to(compute_dtype)
        r2 = row2.to(compute_dtype)
        r3 = row3.to(compute_dtype)
        # Both sub-systems include row2 (from camera 2's x-projection equation),
        # which carries the camera-2 translation.  Using rows {0,1,2} and {1,2,3}
        # rather than {0,1,2} and {0,1,3} keeps at least one sub-system of full
        # rank when camera 2 has zero last-column entries in its y- and
        # z-projection rows (e.g. [R|t] with t = (-T,0,0)): rows 1 and 3 then
        # coincide for a shared K, so {1,2,3} is rank-deficient and {0,1,2} is not.
        A_012 = torch.stack([r0, r1, r2], dim=-2)  # (*, N, 3, 4)
        A_123 = torch.stack([r1, r2, r3], dim=-2)  # (*, N, 3, 4)
        h_012 = null_vector_3x4(A_012)  # (*, N, 4)
        h_123 = null_vector_3x4(A_123)  # (*, N, 4)
        # A rank-deficient sub-system (zero baseline, coinciding rows) has cofactors at roundoff. The cofactors
        # of a full-rank sub-system scale with the product of its row norms, so judge each null vector against
        # that product and use only the full-rank sub-system(s); the point is NaN when neither is.
        tol = 8.0 * torch.finfo(compute_dtype).eps
        n012 = h_012.norm(dim=-1, keepdim=True)
        n123 = h_123.norm(dim=-1, keepdim=True)
        ok012 = n012 > tol * A_012.norm(dim=-1).prod(dim=-1, keepdim=True)
        ok123 = n123 > tol * A_123.norm(dim=-1).prod(dim=-1, keepdim=True)
        v012 = h_012 / torch.where(ok012, n012, torch.ones_like(n012))
        v123 = h_123 / torch.where(ok123, n123, torch.ones_like(n123))
        # Null vectors are defined up to a global sign; align signs before
        # averaging in homogeneous space to prevent cancellation when the two
        # sub-system solutions point in opposite directions (which would yield a
        # near-zero homogeneous vector and NaN after dehomogenisation).
        dot = (v012 * v123).sum(dim=-1, keepdim=True)
        v123 = torch.where(dot < 0, -v123, v123)
        zeros = torch.zeros_like(v012)
        points3d_h = torch.where(ok012, v012, zeros) + torch.where(ok123, v123, zeros)  # (*, N, 4)
        points3d_h = torch.where(ok012 | ok123, points3d_h, torch.full_like(points3d_h, float("nan")))
        points3d_h = points3d_h.to(row0.dtype)

    else:
        raise NotImplementedError(f"Unknown solver '{solver}'. Choose from: 'svd', 'eigh', 'cofactor'.")

    # A correspondence at infinity (parallel rays) has w = 0 up to roundoff: the rows carry the input dtype's
    # roundoff and the solver the compute dtype's, which eigh amplifies by squaring the conditioning (about 25
    # epsilons on the two-view fixture against 2 for svd). Flag it as NaN instead of dividing by that roundoff.
    w_scale = 128.0 if solver == "eigh" else 16.0
    w_tol = w_scale * torch.finfo(compute_dtype).eps + 4.0 * torch.finfo(points3d_h.dtype).eps
    at_infinity = points3d_h[..., 3:].abs() <= w_tol * points3d_h.norm(dim=-1, keepdim=True)
    points3d_h = torch.where(at_infinity, torch.full_like(points3d_h, float("nan")), points3d_h)

    points3d: torch.Tensor = convert_points_from_homogeneous(points3d_h)
    return points3d
