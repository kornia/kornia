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

"""Module containing the functionalities for computing the Fundamental Matrix."""

import math
from typing import Literal, Optional, Tuple

import torch

from kornia.core import Tensor, concatenate, ones_like, where, zeros
from kornia.core.check import KORNIA_CHECK_SAME_SHAPE, KORNIA_CHECK_SHAPE
from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from kornia.geometry.solvers import solve_cubic
from kornia.utils.helpers import _torch_svd_cast, safe_inverse_with_mask


def normalize_points(points: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    r"""Normalize points (isotropic).

    Computes the transformation matrix such that the two principal moments of the set of points
    are equal to unity, forming an approximately symmetric circular cloud of points of radius 1
    about the origin. Reference: Hartley/Zisserman 4.4.4 pag.107

    This operation is an essential step before applying the DLT algorithm in order to consider
    the result as optimal.

    Args:
       points: Tensor containing the points to be normalized with shape :math:`(B, N, 2)`.
       eps: epsilon value to avoid numerical instabilities.

    Returns:
       tuple containing the normalized points in the shape :math:`(B, N, 2)` and the transformation matrix
       in the shape :math:`(B, 3, 3)`.

    """
    if points.ndim != 3:
        raise AssertionError(points.shape)
    if points.shape[-1] != 2:
        raise AssertionError(points.shape)

    B, N, _ = points.shape
    device, dtype = points.device, points.dtype

    # Center at mean
    x_mean = points.mean(dim=1, keepdim=True)  # (B,1,2)
    centered = points - x_mean  # (B,N,2)

    # Mean Euclidean distance to origin (radius)
    mean_radius = centered.norm(dim=-1, p=2).mean(dim=-1)  # (B,)

    # Scale so that mean radius becomes sqrt(2)
    scale = (math.sqrt(2.0)) / (mean_radius + eps)  # (B,)

    # Apply similarity transform in-place-ish (broadcast scale)
    points_norm = centered * scale.view(B, 1, 1)  # (B,N,2)

    # Build transform matrix:
    # T = [[s, 0, -s*mx],
    #      [0, s, -s*my],
    #      [0, 0,   1  ]]
    transform = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    transform[..., 0, 0] = scale
    transform[..., 1, 1] = scale
    transform[..., 0, 2] = -scale * x_mean[..., 0, 0]
    transform[..., 1, 2] = -scale * x_mean[..., 0, 1]
    transform[..., 2, 2] = 1.0

    return points_norm, transform


def normalize_transformation(M: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Normalize a given transformation matrix.

    The function trakes the transformation matrix and normalize so that the value in
    the last row and column is one.

    Args:
        M: The transformation to be normalized of any shape with a minimum size of 2x2.
        eps: small value to avoid unstabilities during the backpropagation.

    Returns:
        the normalized transformation matrix with same shape as the input.

    """
    if len(M.shape) < 2:
        raise AssertionError(M.shape)
    norm_val: Tensor = M[..., -1:, -1:]
    return where(norm_val.abs() > eps, M / (norm_val + eps), M)


def _nullspace_via_eigh(A: torch.Tensor) -> torch.Tensor:
    """A: (..., 7, 9)
    Returns N: (..., 9, 2) where columns span the right nullspace of A
    """
    AT = A.transpose(-2, -1)  # (..., 9, 7)
    G = AT @ A  # (..., 9, 9) SPD
    evals, evecs = torch.linalg.eigh(G)  # ascending eigenvalues
    N = evecs[..., :, :2]  # eigenvectors for 2 smallest evals
    return N  # orthonormal columns


def _F1F2_from_nullspace(N: torch.Tensor):
    """N: (..., 9, 2)
    Returns F1, F2: (..., 3, 3)
    """
    F1 = N[..., 0].view(-1, 3, 3)
    F2 = N[..., 1].view(-1, 3, 3)
    return F1, F2


def _normalize_F(F: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Frobenius-normalize each 3x3 (keeps cubic coefficients well-scaled)."""
    nrm = F.abs().sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return F / nrm


# Reference: Adapted from the 'run_7point' function in opencv
# https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/fundam.cpp
@torch.jit.script
def run_7point(points1: Tensor, points2: Tensor) -> Tensor:
    r"""Compute the fundamental matrix using the 7-point algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3*m, 3), Valid values of m are 1, 2 or 3`

    """
    KORNIA_CHECK_SHAPE(points1, ["B", "7", "2"])
    KORNIA_CHECK_SHAPE(points2, ["B", "7", "2"])

    batch_size = points1.shape[0]

    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # Bx1xN

    ones = ones_like(x1)
    # form a linear system: which represents
    # the equation (x2[i], 1)*F*(x1[i], 1) = 0
    X = concatenate([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], -1)  # BxNx9

    # X * Fmat = 0 is singular (7 equations for 9 variables)
    # solving for nullspace of X to get two F
    # Slower original SVD route
    # _, _, v = _torch_svd_cast(X)
    # last two singular vector as a basic of the space
    # f1 = v[..., 7].view(-1, 3, 3)
    # f2 = v[..., 8].view(-1, 3, 3)
    f1, f2 = _F1F2_from_nullspace(_nullspace_via_eigh(X))
    f1, f2 = _normalize_F(f1), _normalize_F(f2)

    # lambda*f1 + mu*f2 is an arbitrary fundamental matrix
    # f ~ lambda*f1 + (1 - lambda)*f2
    # det(f) = det(lambda*f1 + (1-lambda)*f2), find lambda
    # form a cubic equation
    # finding the coefficients of cubic polynomial (coeffs)

    coeffs = zeros((batch_size, 4), device=f1.device, dtype=f1.dtype)
    f1_det = torch.linalg.det(f1)
    f2_det = torch.linalg.det(f2)
    coeffs[:, 0] = f1_det
    coeffs[:, 1] = torch.einsum("bii->b", f2 @ safe_inverse_with_mask(f1)[0]) * f1_det
    coeffs[:, 2] = torch.einsum("bii->b", f1 @ safe_inverse_with_mask(f2)[0]) * f2_det
    coeffs[:, 3] = f2_det

    # solve the cubic equation, there can be 1 to 3 roots
    # roots = torch.tensor(np.roots(coeffs.numpy()))
    roots = solve_cubic(coeffs)

    fmatrix = zeros((batch_size, 3, 3, 3), device=f1.device, dtype=f1.dtype)
    cnz = torch.count_nonzero(roots, dim=1)
    valid_root_mask = (cnz < 3) | (cnz > 1)

    _lambda = roots
    _mu = torch.ones_like(_lambda)

    _s = f1[valid_root_mask, 2, 2].unsqueeze(dim=1) * roots[valid_root_mask] + f2[valid_root_mask, 2, 2].unsqueeze(
        dim=1
    )
    _s_non_zero_mask = ~torch.isclose(_s, torch.tensor(0.0, device=f1.device, dtype=f1.dtype))

    _mu[_s_non_zero_mask] = 1.0 / _s[_s_non_zero_mask]
    _lambda[_s_non_zero_mask] = _lambda[_s_non_zero_mask] * _mu[_s_non_zero_mask]

    f1_expanded = f1.unsqueeze(1).expand(batch_size, 3, 3, 3)
    f2_expanded = f2.unsqueeze(1).expand(batch_size, 3, 3, 3)

    fmatrix[valid_root_mask] = (
        f1_expanded[valid_root_mask] * _lambda[valid_root_mask, :, None, None]
        + f2_expanded[valid_root_mask] * _mu[valid_root_mask, :, None, None]
    )

    mat_ind = zeros(3, 3, dtype=torch.bool)
    mat_ind[2, 2] = True
    fmatrix[_s_non_zero_mask, mat_ind] = 1.0
    fmatrix[~_s_non_zero_mask, mat_ind] = 0.0

    trans1_exp = transform1[valid_root_mask].unsqueeze(1).expand(-1, fmatrix.shape[2], -1, -1)
    trans2_exp = transform2[valid_root_mask].unsqueeze(1).expand(-1, fmatrix.shape[2], -1, -1)

    fmatrix[valid_root_mask] = torch.matmul(
        trans2_exp.transpose(-2, -1), torch.matmul(fmatrix[valid_root_mask], trans1_exp)
    )
    return normalize_transformation(fmatrix)


@torch.jit.script
def run_8point(points1: Tensor, points2: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    r"""Compute the fundamental matrix using (weighted) 8-point DLT, optimized.

    Args:
        points1: (B, N, 2), N >= 8
        points2: (B, N, 2), N >= 8
        weights: optional (B, N) nonnegative weights

    Returns:
        (B, 3, 3) fundamental matrices
    """
    small_n_threshold: int = 512
    KORNIA_CHECK_SHAPE(points1, ["B", "N", "2"])
    KORNIA_CHECK_SHAPE(points2, ["B", "N", "2"])
    KORNIA_CHECK_SAME_SHAPE(points1, points2)
    if points1.shape[1] < 8:
        raise AssertionError(points1.shape)
    if weights is not None:
        KORNIA_CHECK_SHAPE(weights, ["B", "N"])
        if weights.shape[1] != points1.shape[1]:
            raise AssertionError(weights.shape)

    # Hartley normalization (same as before)
    pts1n, T1 = normalize_points(points1)
    pts2n, T2 = normalize_points(points2)

    x1, y1 = torch.chunk(pts1n, dim=-1, chunks=2)  # (B,N,1)
    x2, y2 = torch.chunk(pts2n, dim=-1, chunks=2)  # (B,N,1)
    ones = torch.ones_like(x1)

    # Design matrix rows A_i = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    # Shape: A ∈ (B, N, 9)
    A = torch.cat([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], dim=-1).squeeze(-2)

    B, N, _ = A.shape
    device, dtype = A.device, A.dtype

    # Build normal matrix M = A^T W A  (B,9,9) without forming NxN diagonals.
    if weights is None:
        if N < small_n_threshold:
            # Use GEMM on tall A: (B,9,N) @ (B,N,9)
            M = A.transpose(-2, -1).contiguous() @ A
        else:
            # Accumulate via einsum (saves bandwidth for huge N)
            M = torch.einsum("bni,bnj->bij", A, A)
    else:
        w = weights.clamp_min(0)
        if N < small_n_threshold:
            # Row-scale by sqrt(w) then GEMM
            Aw = A * w.unsqueeze(-1).sqrt()
            M = Aw.transpose(-2, -1).contiguous() @ Aw
        else:
            # Weighted einsum
            M = torch.einsum("bni,bnj,bn->bij", A, A, w)

    evals, evecs = torch.linalg.eigh(M)  # ascending order
    h = evecs[..., 0]  # (B,9), eigenvector for smallest λ
    F_hat = h.view(B, 3, 3)

    # Enforce rank-2 with a 3x3 SVD
    U, S, V = _torch_svd_cast(F_hat)
    S_new = torch.zeros_like(S)
    S_new[..., :-1] = S[..., :-1]
    F_rank2 = U @ torch.diag_embed(S_new) @ V.mH
    F = T2.transpose(-2, -1) @ (F_rank2 @ T1)

    return normalize_transformation(F)


def find_fundamental(
    points1: Tensor, points2: Tensor, weights: Optional[Tensor] = None, method: Literal["8POINT", "7POINT"] = "8POINT"
) -> Tensor:
    r"""Find the fundamental matrix.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
        method: The method to use for computing the fundamental matrix. Supported methods are "7POINT" and "8POINT".

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3*m, 3)`, where `m` number of fundamental matrix.

    Raises:
        ValueError: If an invalid method is provided.

    """
    if method.upper() == "7POINT":
        result = run_7point(points1, points2)
    elif method.upper() == "8POINT":
        result = run_8point(points1, points2, weights)
    else:
        raise ValueError(f"Invalid method: {method}. Supported methods are '7POINT' and '8POINT'.")
    return result


def compute_correspond_epilines(points: Tensor, F_mat: Tensor) -> Tensor:
    r"""Compute the corresponding epipolar line for a given set of points.

    Args:
        points: tensor containing the set of points to project in the shape of :math:`(*, N, 2)` or :math:`(*, N, 3)`.
        F_mat: the fundamental to use for projection the points in the shape of :math:`(*, 3, 3)`.

    Returns:
        a tensor with shape :math:`(*, N, 3)` containing a vector of the epipolar
        lines corresponding to the points to the other image. Each line is described as
        :math:`ax + by + c = 0` and encoding the vectors as :math:`(a, b, c)`.

    """
    KORNIA_CHECK_SHAPE(points, ["*", "N", "DIM"])
    if points.shape[-1] == 2:
        points_h: Tensor = convert_points_to_homogeneous(points)
    elif points.shape[-1] == 3:
        points_h = points
    else:
        raise AssertionError(points.shape)
    KORNIA_CHECK_SHAPE(F_mat, ["*", "3", "3"])
    # project points and retrieve lines components
    points_h = torch.transpose(points_h, dim0=-2, dim1=-1)
    a, b, c = torch.chunk(F_mat @ points_h, dim=-2, chunks=3)

    # compute normal and compose equation line
    nu: Tensor = a * a + b * b
    nu = where(nu > 0.0, 1.0 / torch.sqrt(nu), torch.ones_like(nu))

    line = torch.cat([a * nu, b * nu, c * nu], dim=-2)  # *x3xN
    return torch.transpose(line, dim0=-2, dim1=-1)  # *xNx3


def get_perpendicular(lines: Tensor, points: Tensor) -> Tensor:
    r"""Compute the perpendicular to a line, through the point.

    Args:
        lines: tensor containing the set of lines :math:`(*, N, 3)`.
        points:  tensor containing the set of points :math:`(*, N, 2)`.

    Returns:
        a tensor with shape :math:`(*, N, 3)` containing a vector of the epipolar
        perpendicular lines. Each line is described as
        :math:`ax + by + c = 0` and encoding the vectors as :math:`(a, b, c)`.

    """
    KORNIA_CHECK_SHAPE(lines, ["*", "N", "3"])
    KORNIA_CHECK_SHAPE(points, ["*", "N", "two"])
    if points.shape[2] == 2:
        points_h: Tensor = convert_points_to_homogeneous(points)
    elif points.shape[2] == 3:
        points_h = points
    else:
        raise AssertionError(points.shape)
    infinity_point = lines * torch.tensor([1, 1, 0], dtype=lines.dtype, device=lines.device).view(1, 1, 3)
    perp: Tensor = torch.linalg.cross(points_h, infinity_point, dim=2)
    return perp


def get_closest_point_on_epipolar_line(pts1: Tensor, pts2: Tensor, Fm: Tensor) -> Tensor:
    """Return closest point on the epipolar line to the correspondence, given the fundamental matrix.

    Args:
        pts1: correspondences from the left images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        pts2: correspondences from the right images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.

    Returns:
        point on epipolar line :math:`(*, N, 2)`.

    """
    if not isinstance(Fm, Tensor):
        raise TypeError(f"Fm type is not a torch.Tensor. Got {type(Fm)}")
    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")
    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)
    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)
    line1in2 = compute_correspond_epilines(pts1, Fm)
    perp = get_perpendicular(line1in2, pts2)
    points1_in_2 = convert_points_from_homogeneous(torch.linalg.cross(line1in2, perp, dim=2))

    return points1_in_2


def fundamental_from_essential(E_mat: Tensor, K1: Tensor, K2: Tensor) -> Tensor:
    r"""Get the Fundamental matrix from Essential and camera matrices.

    Uses the method from Hartley/Zisserman 9.6 pag 257 (formula 9.12).

    Args:
        E_mat: The essential matrix with shape of :math:`(*, 3, 3)`.
        K1: The camera matrix from first camera with shape :math:`(*, 3, 3)`.
        K2: The camera matrix from second camera with shape :math:`(*, 3, 3)`.

    Returns:
        The fundamental matrix with shape :math:`(*, 3, 3)`.

    """
    KORNIA_CHECK_SHAPE(E_mat, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(K1, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(K2, ["*", "3", "3"])
    if not len(E_mat.shape[:-2]) == len(K1.shape[:-2]) == len(K2.shape[:-2]):
        raise AssertionError

    return (safe_inverse_with_mask(K2)[0]).transpose(-2, -1) @ E_mat @ (safe_inverse_with_mask(K1)[0])


# adapted from:
# https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/fundamental.cpp#L109
# https://github.com/openMVG/openMVG/blob/160643be515007580086650f2ae7f1a42d32e9fb/src/openMVG/multiview/projection.cpp#L134


def fundamental_from_projections(P1: Tensor, P2: Tensor) -> Tensor:
    r"""Get the Fundamental matrix from Projection matrices.

    Args:
        P1: The projection matrix from first camera with shape :math:`(*, 3, 4)`.
        P2: The projection matrix from second camera with shape :math:`(*, 3, 4)`.

    Returns:
         The fundamental matrix with shape :math:`(*, 3, 3)`.

    """
    KORNIA_CHECK_SHAPE(P1, ["*", "3", "4"])
    KORNIA_CHECK_SHAPE(P2, ["*", "3", "4"])
    if P1.shape[:-2] != P2.shape[:-2]:
        raise AssertionError

    def vstack(x: Tensor, y: Tensor) -> Tensor:
        return concatenate([x, y], dim=-2)

    input_dtype = P1.dtype
    if input_dtype not in (torch.float32, torch.float64):
        P1 = P1.to(torch.float32)
        P2 = P2.to(torch.float32)

    X1 = P1[..., 1:, :]
    X2 = vstack(P1[..., 2:3, :], P1[..., 0:1, :])
    X3 = P1[..., :2, :]

    Y1 = P2[..., 1:, :]
    Y2 = vstack(P2[..., 2:3, :], P2[..., 0:1, :])
    Y3 = P2[..., :2, :]

    X1Y1, X2Y1, X3Y1 = vstack(X1, Y1), vstack(X2, Y1), vstack(X3, Y1)
    X1Y2, X2Y2, X3Y2 = vstack(X1, Y2), vstack(X2, Y2), vstack(X3, Y2)
    X1Y3, X2Y3, X3Y3 = vstack(X1, Y3), vstack(X2, Y3), vstack(X3, Y3)

    F_vec = torch.cat(
        [
            X1Y1.det().reshape(-1, 1),
            X2Y1.det().reshape(-1, 1),
            X3Y1.det().reshape(-1, 1),
            X1Y2.det().reshape(-1, 1),
            X2Y2.det().reshape(-1, 1),
            X3Y2.det().reshape(-1, 1),
            X1Y3.det().reshape(-1, 1),
            X2Y3.det().reshape(-1, 1),
            X3Y3.det().reshape(-1, 1),
        ],
        dim=1,
    )

    return F_vec.view(*P1.shape[:-2], 3, 3).to(input_dtype)
