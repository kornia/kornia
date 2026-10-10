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


def _sub_system_null_vectors(A: torch.Tensor) -> torch.Tensor:
    r"""Return the null vectors of the four :math:`3 \times 4` sub-systems of each :math:`4 \times 4` matrix.

    Row ``j`` of the result is orthogonal to every row of ``A`` but row ``3 - j`` (sub-systems ``{0, 1, 2}``,
    ``{0, 1, 3}``, ``{0, 2, 3}``, ``{1, 2, 3}``); up to sign it is the column of the adjugate of ``A`` for the
    row left out, the vector of :math:`3 \times 3` minors that :func:`~kornia.geometry.solvers.null_vector_3x4`
    returns. The minors are expanded along the ``2 x 2`` minors of the row pairs ``(0, 1)`` and ``(2, 3)``, which
    all four share: the Hodge dual of one pair's bivector applied to each row of the other pair. These are the
    columns of :func:`kornia.core._small_linalg._adjugate_4x4` in fewer, wider kernels, about half its time for a
    million matrices on the CPU and on MPS.

    Args:
        A: batch of matrices, shape ``(*, 4, 4)``.

    Returns:
        The four null vectors, shape ``(*, 4, 4)``, not normalised: a sub-system of rank below 3 gives zero.
    """
    P = A[..., 0::2, None, :]  # rows 0 and 2, (*, 2, 1, 4)
    Q = A[..., 1::2, None, :]  # rows 1 and 3

    def minor(p: int, q: int) -> torch.Tensor:
        return P[..., p] * Q[..., q] - P[..., q] * Q[..., p]  # (*, 2, 1)

    s01, s02, s03, s12, s13, s23 = minor(0, 1), minor(0, 2), minor(0, 3), minor(1, 2), minor(1, 3), minor(2, 3)
    # The bivector of rows (0, 1) meets rows 2 and 3; that of rows (2, 3) meets rows 0 and 1.
    x0, x1, x2, x3 = torch.stack([A[..., 2:, :], A[..., :2, :]], dim=-3).unbind(-1)  # each (*, 2, 2)
    h = torch.stack(
        [
            s23 * x1 - s13 * x2 + s12 * x3,
            -s23 * x0 + s03 * x2 - s02 * x3,
            s13 * x0 - s03 * x1 + s01 * x3,
            -s12 * x0 + s02 * x1 - s01 * x2,
        ],
        dim=-1,
    )  # (*, 2, 2, 4)
    return h.flatten(-3, -2)


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
          scale of ``t``. Mask those rows before a loss (``out[~out.isnan().any(-1)]``); the gradients with respect
          to the cameras and the other points stay finite.
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
          * ``"cofactor"`` — closed form, no LAPACK call. The null vectors of the four
            :math:`3 \times 4` sub-systems (the cofactors, as in
            :func:`~kornia.geometry.solvers.null_vector_3x4`) form the adjugate of the
            DLT matrix; the longest one, refined by one step of inverse iteration through
            the adjugate, gives the ``"svd"`` point to roundoff without noise and nearly
            the same point with noise, also for rectified and vertical stereo pairs, where
            a sub-system is nearly rank-deficient. When the noise is comparable to the
            parallax, so that the DLT point itself is far off, one step does not converge
            and the two can differ. The point is NaN when every sub-system is
            rank-deficient (zero baseline). Fastest option for large batches.

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
        # The null vectors of the four 3x4 sub-systems (each DLT row left out once) are, up to sign, the columns
        # of adj(A) = det(A) A^-1 = det(A) V S^-1 U^T. Each leans towards the DLT solution v4 (the svd one), and
        # all are parallel to it when A has rank 3 (noise-free). With noise, a sub-system whose three rows are
        # nearly dependent (a rectified or vertical stereo pair, a camera rolled by 90 degrees, a row through the
        # epipole) has a null vector set by the noise, and no fixed choice or average of two of them avoids it.
        # So take the longest one and apply one step of inverse iteration with
        # adj(A) adj(A)^T = det(A)^2 (A^T A)^-1, which suppresses v3 against v4 by a further (s4 / s3)^2: the
        # svd solution without a LAPACK call, still defined at rank 3, where A^-1 is not.
        # The minors use only arithmetic ops, so promote fp16/bf16 -> fp32 and stay there until the null vector
        # is normalised: the unnormalised minors of pixel-scale rows overflow float16.
        compute_dtype = _normalize_to_float32_or_float64(row0.dtype)
        tiny = torch.finfo(compute_dtype).tiny
        A = torch.stack([row0, row1, row2, row3], dim=-2).to(compute_dtype)  # (*, N, 4, 4)
        # A common scale of the rows leaves the solution unchanged and keeps the cubic minors in range.
        row_norms = A.norm(dim=-1)  # (*, N, 4)
        scale = row_norms.amax(dim=-1, keepdim=True).clamp_min(tiny)
        A, row_norms = A / scale[..., None], row_norms / scale
        H = _sub_system_null_vectors(A)  # (*, N, 4, 4); row j leaves out DLT row 3 - j
        # A sub-system of rank below 3 has minors at roundoff, and every one has when A has rank 2 (zero
        # baseline): judge each against the product of its row norms, and return NaN when none has full rank.
        h_norms = H.norm(dim=-1)  # (*, N, 4)
        sub_row_norms = row_norms.prod(dim=-1, keepdim=True) / row_norms.flip(-1).clamp_min(tiny)
        full_rank = (h_norms > 8.0 * torch.finfo(compute_dtype).eps * sub_row_norms).any(dim=-1, keepdim=True)
        longest = h_norms.argmax(dim=-1, keepdim=True)[..., None].expand(*H.shape[:-2], 1, 4)
        c = H.gather(-2, longest).squeeze(-2) / h_norms.amax(dim=-1, keepdim=True).clamp_min(tiny)
        w = (H @ c[..., None]).squeeze(-1)  # adj(A)^T c, up to the signs of its entries
        w = w / w.norm(dim=-1, keepdim=True).clamp_min(tiny)
        v = (H.mT @ w[..., None]).squeeze(-1)  # adj(A) adj(A)^T c: the signs cancel
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(tiny)
        points3d_h = torch.where(full_rank, v, torch.full_like(v, float("nan"))).to(row0.dtype)

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
