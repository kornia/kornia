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
from typing import Literal, Optional, Tuple, Union

import torch

from kornia.core.check import KORNIA_CHECK_SAME_SHAPE, KORNIA_CHECK_SHAPE
from kornia.core.utils import _torch_svd_cast, safe_inverse_with_mask
from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from kornia.geometry.solvers.homogeneous import _det3, _null_space_lu
from kornia.geometry.solvers.polynomial_solver import _solve_cubic_real


def normalize_points(
    points: torch.Tensor, eps: float = 1e-8, weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Normalize points (isotropic).

    Computes the Hartley normalisation: the points are translated to zero mean and scaled isotropically so
    that their mean distance to the origin is :math:`\sqrt{2}`. Reference: Hartley/Zisserman 4.4.4 pag.107

    This operation is an essential step before applying the DLT algorithm in order to consider
    the result as optimal.

    Args:
       points: Tensor containing the points to be normalized with shape :math:`(B, N, 2)`.
       eps: epsilon value to avoid numerical instabilities.
       weights: Optional nonnegative weights with shape :math:`(B, N)` for the centroid and mean radius.
          Zero-weight points do not influence the transform. An all-zero batch element uses unweighted statistics.

    Returns:
       tuple containing the normalized points in the shape :math:`(B, N, 2)` and the transformation matrix
       in the shape :math:`(B, 3, 3)` that maps the input points to them.

    """
    if points.ndim != 3:
        raise AssertionError(points.shape)
    if points.shape[-1] != 2:
        raise AssertionError(points.shape)

    B, _N, _ = points.shape
    device, dtype = points.device, points.dtype

    if weights is None:
        x_mean = points.mean(dim=1, keepdim=True)  # (B,1,2)
    else:
        if weights.shape != points.shape[:2]:
            raise AssertionError(weights.shape)
        # Accumulate in at least float32: in half precision a sum followed by a division rounds twice where
        # ``mean`` rounds once, and uniform weights would then not reproduce the unweighted statistics.
        acc_dtype = torch.promote_types(dtype, torch.float32)
        # Negative weights count as zero. ``where`` rather than ``clamp_min(0)``: the clamp's derivative at the bound
        # depends on the torch version (#4229), and a weight of exactly 0 is how a correspondence is dropped.
        positive_weights = weights.to(acc_dtype)
        positive_weights = torch.where(positive_weights < 0, 0.0, positive_weights)
        total_weight = positive_weights.sum(dim=1, keepdim=True)
        # A fully de-weighted sample is degenerate; keep its normalization finite and batched.
        effective_weights = torch.where(total_weight > 0, positive_weights, torch.ones_like(positive_weights))
        total_weight = effective_weights.sum(dim=1, keepdim=True)
        weighted_sum = (points.to(acc_dtype) * effective_weights[..., None]).sum(dim=1, keepdim=True)
        x_mean = (weighted_sum / total_weight[..., None]).to(dtype)
    centered = points - x_mean  # (B,N,2)

    # Mean Euclidean distance to origin (radius)
    radii = centered.norm(dim=-1, p=2)
    if weights is None:
        mean_radius = radii.mean(dim=-1)  # (B,)
    else:
        mean_radius = ((radii.to(acc_dtype) * effective_weights).sum(dim=-1) / total_weight.squeeze(-1)).to(dtype)

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


def normalize_transformation(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Normalize a given transformation matrix.

    Convention:
        - Divides ``M`` by its last entry ``M[..., -1, -1]``, which gives :func:`find_fundamental` its
          ``F[2, 2] = 1`` scaling. A matrix whose last entry is within ``eps`` of zero is returned unchanged.

    Args:
        M: The transformation to be normalized of any shape with a minimum size of 2x2.
        eps: magnitude of the last entry at or below which ``M`` is returned unchanged.

    Returns:
        the normalized transformation matrix with same shape as the input.

    """
    if len(M.shape) < 2:
        raise AssertionError(M.shape)
    norm_val: torch.Tensor = M[..., -1:, -1:]
    mask = norm_val.abs() > eps
    divisor = torch.where(mask, norm_val, torch.ones_like(norm_val))
    return torch.where(mask, M / divisor, M)


def _epipolar_design_rows(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Rows ``vec(x2 x1^T)`` of the epipolar constraint, so that ``row . vec(F) = x2^T F x1`` with ``F`` row-major.

    Each entry is one product ``x2_i x1_j``, so the constructions below give the same bits; the cheapest depends on
    the device. On CPU the broadcast outer product costs 2.5x the column-wise form (256 samples x 100 points: 402 vs
    157 us); on CUDA it is one kernel instead of a dozen (12 vs 51 us).

    Args:
        x1: points of the first image, homogeneous ``(..., N, 3)`` or inhomogeneous ``(..., N, 2)`` with ``w = 1``.
        x2: points of the second image, in the same form as ``x1``.

    Returns:
        the design matrix ``(..., N, 9)``: ``[x2 x1, x2 y1, x2, y2 x1, y2 y1, y2, x1, y1, 1]`` for inhomogeneous
        points.
    """
    if x1.shape[-1] == 2:
        u1, v1 = torch.chunk(x1, dim=-1, chunks=2)
        u2, v2 = torch.chunk(x2, dim=-1, chunks=2)
        return torch.cat([u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, torch.ones_like(u1)], dim=-1)
    if x1.device.type == "cpu":
        return torch.cat([x2[..., 0:1] * x1, x2[..., 1:2] * x1, x2[..., 2:3] * x1], dim=-1)
    return (x2[..., :, None] * x1[..., None, :]).flatten(-2)


def _seven_point_basis(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The two-dimensional null space of seven epipolar constraints ``(B, 7, 9)`` as two ``(B, 3, 3)`` matrices.

    The null space comes from :func:`~kornia.geometry.solvers.homogeneous._null_space_lu`, in at least float32 since
    no backend factorizes half precision.
    """
    solve_dtype = torch.promote_types(A.dtype, torch.float32)
    basis = _null_space_lu(A.to(solve_dtype)).mT.reshape(-1, 2, 3, 3).to(A.dtype)
    return basis[:, 0], basis[:, 1]


def _det_pencil_coefficients(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    r"""Coefficients ``(B, 4)`` of the cubic :math:`\det(x f_1 + f_2)`, highest degree first.

    Expanded by multilinearity in the rows, so neither matrix needs to be invertible: the leading coefficient is
    :math:`\det f_1` and the constant one :math:`\det f_2`.
    """
    a1, a2, a3 = f1[:, 0], f1[:, 1], f1[:, 2]
    b1, b2, b3 = f2[:, 0], f2[:, 1], f2[:, 2]
    crosses = torch.linalg.cross(torch.stack([a2, b2, a2, b2], 1), torch.stack([a3, b3, b3, a3], 1))
    x_aa, x_bb, x_ab = crosses[:, 0], crosses[:, 1], crosses[:, 2] + crosses[:, 3]
    dots = (torch.stack([a1, b1, a1, a1, b1, b1], 1) * torch.stack([x_aa, x_aa, x_ab, x_bb, x_ab, x_bb], 1)).sum(-1)
    return torch.stack([dots[:, 0], dots[:, 1] + dots[:, 2], dots[:, 3] + dots[:, 4], dots[:, 5]], 1)


def _solve_dtype(device: torch.device) -> torch.dtype:
    """The dtype of the small closed-form steps: float64, except on MPS, which has none."""
    return torch.float32 if device.type == "mps" else torch.float64


def _rank2_projection(F: torch.Tensor) -> torch.Tensor:
    r"""The nearest rank-2 matrices ``(B, 3, 3)`` in Frobenius norm, ``F (I - v v^T)`` for a smallest singular vector.

    ``v`` is an eigenvector of ``F^T F`` for its smallest eigenvalue :math:`\lambda_3`, found without an SVD: the
    eigenvalue from the trigonometric solution of the characteristic polynomial, the vector from the best cross
    product of two rows of ``F^T F - \lambda_3 I``. When :math:`\lambda_3` is repeated those rows span one direction or
    none and every cross product vanishes; any unit vector orthogonal to the rows is then a smallest singular vector:
    the cross product of the largest row with the coordinate axis least aligned to it, or ``e_3`` when every row
    vanishes. The nearest rank-2 matrix is not unique there, and this returns one of them.

    Runs in :func:`_solve_dtype` and returns ``F``'s dtype. ``sqrt``, ``acos`` and the normalizations take safe
    substitutes where their derivative is unbounded (#4229), so gradients stay finite.
    """
    dtype = F.dtype
    F = F.to(_solve_dtype(F.device))
    eye = torch.eye(3, dtype=F.dtype, device=F.device)
    M = F.mT @ F
    trace = M.diagonal(dim1=-2, dim2=-1).sum(-1)
    q = trace / 3
    shifted = M - q[:, None, None] * eye
    p2 = shifted.square().sum((-2, -1)) / 6
    spread = p2 > 0
    p = torch.where(spread, p2, torch.ones_like(p2)).sqrt()
    r = _det3(*(shifted / p[:, None, None]).flatten(-2).unbind(-1)) / 2
    inside = r.abs() < 1
    boundary = torch.where(r > 0, torch.zeros_like(r), torch.full_like(r, math.pi))
    angle = torch.where(inside, torch.acos(torch.where(inside, r, torch.zeros_like(r))), boundary) / 3
    smallest = q + torch.where(spread, 2 * p * torch.cos(angle + 2 * math.pi / 3), torch.zeros_like(p))
    rows = M - smallest[:, None, None] * eye
    crosses = torch.linalg.cross(rows[:, [0, 0, 1]], rows[:, [1, 2, 2]])
    cross_norms = crosses.square().sum(-1)
    best = cross_norms.argmax(1, keepdim=True)
    cross = crosses.gather(1, best[..., None].expand(-1, 1, 3))[:, 0]
    cross_norm = cross_norms.gather(1, best)[:, 0]
    row_norms = rows.square().sum(-1)
    top = row_norms.argmax(1, keepdim=True)
    row = rows.gather(1, top[..., None].expand(-1, 1, 3))[:, 0]
    row_norm = row_norms.gather(1, top)[:, 0]
    perpendicular = torch.linalg.cross(row, eye[row.abs().argmin(1)])
    # Rows are rounded at about eps * trace(F^T F); cross products and rows below that carry no direction.
    noise = (8 * torch.finfo(F.dtype).eps * trace).square()
    two_rows = cross_norm > noise * row_norm
    one_row = row_norm > noise
    v = torch.where(two_rows[:, None], cross, torch.where(one_row[:, None], perpendicular, eye[2].expand_as(cross)))
    norm = v.square().sum(-1)
    v = v * torch.where(norm > 0, norm, torch.ones_like(norm)).rsqrt()[:, None]
    return (F - (F @ v[:, :, None]) @ v[:, None, :]).to(dtype)


# The closed-form rank-2 step costs about 45 small kernels whatever the batch: it is faster than a batched 3x3 SVD
# from 128 matrices on CPU (2048: 3.2 -> 0.8 ms) and from 512 on CUDA (2048: 3.2 -> 1.2 ms), slower below
# (i7-14700K / RTX 4090, torch 2.14).
_RANK2_CLOSED_FORM_MIN_BATCH_CPU = 128
_RANK2_CLOSED_FORM_MIN_BATCH_ACCELERATOR = 512


def _enforce_rank2(F: torch.Tensor) -> torch.Tensor:
    """Remove the smallest singular value of ``(B, 3, 3)`` matrices.

    :func:`_rank2_projection` for large batches, an SVD for small ones, where it is cheaper. Without float64 (MPS) the
    closed form would run in float32, where its error grows like ``eps * (sigma_1 / sigma_2)^2`` through ``F^T F``, so
    the SVD is kept for every batch there.
    """
    threshold = _RANK2_CLOSED_FORM_MIN_BATCH_CPU if F.device.type == "cpu" else _RANK2_CLOSED_FORM_MIN_BATCH_ACCELERATOR
    if F.shape[0] >= threshold and _solve_dtype(F.device) == torch.float64:
        return _rank2_projection(F)
    U, S, V = _torch_svd_cast(F)
    S_new = torch.zeros_like(S)
    S_new[..., :-1] = S[..., :-1]
    return U @ torch.diag_embed(S_new) @ V.mH


def _eight_point_null_vector(A: torch.Tensor) -> torch.Tensor:
    """The unit null vectors ``(B, 9)`` of eight epipolar constraints ``A`` ``(B, 8, 9)``, in ``A``'s dtype.

    One batched LU factorization (:func:`~kornia.geometry.solvers.homogeneous._null_space_lu`) finds them much faster
    than ``eigh`` of ``A^T A``, which loops over the batch on CPU and squares the condition number; the factorization
    runs in at least float32 since no backend factorizes half precision.
    """
    h = _null_space_lu(A.to(torch.promote_types(A.dtype, torch.float32)))[..., 0]
    return (h / h.norm(dim=-1, keepdim=True)).to(A.dtype)


def _eight_point_fundamental(A: torch.Tensor) -> torch.Tensor:
    """Rank-2 fundamental matrices ``(B, 3, 3)`` from eight epipolar constraints ``A`` ``(B, 8, 9)``, in ``A``'s dtype.

    :func:`_eight_point_null_vector` followed by :func:`_rank2_projection`: RANSAC's eight-point sampler, which passes
    rows of points it normalized once per call, in batches large enough for the closed form.
    """
    return _rank2_projection(_eight_point_null_vector(A).reshape(-1, 3, 3))


def _seven_point_candidates(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fundamental matrices through seven correspondences, one per real root of ``det(x f_1 + f_2) = 0``.

    The two-dimensional null space ``x f_1 + f_2`` of the constraints ``A`` ``(B, 7, 9)`` (:func:`_epipolar_design_rows`
    of normalized points) is completed by the real roots of its determinant (Hartley and Zisserman, section 11.1.2), so
    every candidate has rank two. The cubic is solved in :func:`_solve_dtype`, choosing its leading matrix from
    ``f_1``, ``f_2``, ``f_1 + f_2`` and ``f_1 - f_2`` to maximize the leading determinant. These four evaluations
    determine the homogeneous cubic, so both singular basis matrices still give a well-conditioned parametrization.

    Returns:
        Candidates ``(B, 3, 3, 3)`` of unit Frobenius norm in ``A``'s dtype, and a mask ``(B, 3)`` of the real roots
        with finite candidates. A cubic with one real root repeats its candidate in the two masked slots rather than
        producing NaN, so that masking them afterwards leaves finite gradients.
    """
    solve_dtype = _solve_dtype(A.device)
    f1, f2 = (f.to(solve_dtype) for f in _seven_point_basis(A))
    coefficients = _det_pencil_coefficients(f1, f2)
    a, b, c, d = coefficients.unbind(1)
    # For lead = f1 +/- f2 and rest = f2, substitute (x, +/-x + 1) into the homogeneous cubic.
    plus = torch.stack([a + b + c + d, b + 2 * c + 3 * d, c + 3 * d, d], 1)
    minus = torch.stack([a - b + c - d, b - 2 * c + 3 * d, c - 3 * d, d], 1)
    choices = torch.stack([coefficients, coefficients.flip(1), plus, minus], 1)
    best = choices[:, :, 0].abs().argmax(1)
    coefficients = choices.gather(1, best[:, None, None].expand(-1, 1, 4))[:, 0]
    directions = torch.stack([f1, f2, f1 + f2, f1 - f2], 1)
    lead = directions.gather(1, best[:, None, None, None].expand(-1, 1, 3, 3))[:, 0]
    rest = torch.where((best == 1)[:, None, None], f1, f2)
    # An identically zero determinant has no isolated roots. Substitute a finite cubic before masking the row.
    isolated = coefficients[:, 0] != 0
    fallback = coefficients.new_tensor([1.0, 0.0, 0.0, 0.0])
    roots, valid = _solve_cubic_real(torch.where(isolated[:, None], coefficients, fallback))
    valid = valid & isolated[:, None]
    F = roots[:, :, None, None] * lead[:, None] + rest[:, None]
    F = F * F.square().sum((-2, -1), keepdim=True).rsqrt()
    return F.to(A.dtype), valid & torch.isfinite(F).flatten(-2).all(-1)


def _robust_loss(r2: torch.Tensor, loss: str, scale2: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """IRLS weights and costs of a squared residual: Cauchy, or truncated at ``scale2``.

    ``scale2`` may be a 0-d tensor, which keeps a compiled caller free of a host synchronization.
    """
    if loss == "cauchy":
        return 1.0 / (1.0 + r2 / scale2), torch.log1p(r2 / scale2)
    return (r2 < scale2).to(r2.dtype), torch.fmin(r2, torch.as_tensor(scale2, dtype=r2.dtype, device=r2.device))


def _hat_basis(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """``E[a] = [e_a]_x``, the generators of rotations, as ``(3, 3, 3)``."""
    return torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=dtype,
        device=device,
    )


def _sampson_cost(
    F: torch.Tensor,
    algebraic: torch.Tensor,
    quadratic: torch.Tensor,
    mask: Optional[torch.Tensor],
    loss: str,
    scale2: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Robust Sampson costs without constructing trial Jacobians or normal equations."""
    quad1 = F[:, :2, :].mT @ F[:, :2, :]
    quad2 = F[:, :, :2] @ F[:, :, :2].mT
    numerator = F.flatten(1) @ algebraic
    denominator = torch.cat([quad1, quad2], 1).flatten(1) @ quadratic
    if mask is not None:
        # A zero mask excludes the correspondence. Guarding its divisor keeps the residual of a finite row finite, so
        # the zero weight below removes it exactly; an infinite residual times zero would be NaN, also in backward.
        denominator = torch.where(mask != 0, denominator, torch.ones_like(denominator))
    r2 = (numerator * denominator.rsqrt()).square()
    rho = (
        torch.log1p(r2 / scale2)
        if loss == "cauchy"
        else torch.fmin(r2, torch.as_tensor(scale2, dtype=r2.dtype, device=r2.device))
    )
    if mask is not None:
        rho = rho * mask
    return rho.sum(1)


def _sampson_normal_equations(
    F: torch.Tensor,
    tangent: torch.Tensor,
    algebraic: torch.Tensor,
    quadratic: torch.Tensor,
    mask: Optional[torch.Tensor],
    loss: str,
    scale2: Union[float, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robust Gauss-Newton normal equations ``[J^T W J | J^T W r]`` ``(K, P, P + 1)`` and costs ``(K,)``.

    For the Sampson residuals of ``F`` ``(K, 3, 3)``, whose derivatives along ``P`` parameters are ``tangent``
    ``(K, P, 3, 3)``. ``algebraic`` ``(9, N)`` and ``quadratic`` ``(18, N)`` are the per-correspondence monomials of
    the epipolar constraint and of the squared gradient norm, from :func:`_epipolar_design_rows`.
    """
    K, P = tangent.shape[:2]
    stacked = torch.cat([F[:, None], tangent], 1)  # (K, P + 1, 3, 3): F, then the P directions
    # x1^T (F[:2]^T X[:2]) x1 + x2^T (F[:, :2] X[:, :2]^T) x2 is half the derivative of the squared gradient norm.
    quad1 = F[:, None, :2, :].mT @ stacked[:, :, :2, :]
    quad2 = F[:, None, :, :2] @ stacked[:, :, :, :2].mT
    out_c = stacked.reshape(K, P + 1, 9) @ algebraic
    out_g = torch.cat([quad1, quad2], 2).reshape(K, P + 1, 18) @ quadratic
    denominator = out_g[:, 0]
    if mask is not None:
        # See _sampson_cost. Only the divisor is guarded: masking the residual and the whole (K, P, N) Jacobian as
        # well adds nothing to a zero weight and costs the compiled program time.
        denominator = torch.where(mask != 0, denominator, torch.ones_like(denominator))
    inv = denominator.rsqrt()
    r = out_c[:, 0] * inv
    J = (out_c[:, 1:] - (r * inv)[:, None] * out_g[:, 1:]) * inv[:, None]  # (K, P, N)
    w, rho = _robust_loss(r * r, loss, scale2)
    if mask is not None:
        w, rho = w * mask, rho * mask
    Jw = J * w[:, None]
    return torch.cat([Jw @ J.mT, Jw @ r[..., None]], 2), rho.sum(1)


def _refine_fundamental_lm(
    F: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    mask: Optional[torch.Tensor],
    loss: str,
    scale2: Union[float, torch.Tensor],
    iters: int,
) -> torch.Tensor:
    """Levenberg-Marquardt on the Sampson distance, batched over fundamental matrices ``(K, 3, 3)``.

    In the spirit of PoseLib's ``refine_fundamental`` (Larsson and contributors, https://github.com/PoseLib/PoseLib):
    ``F = U diag(1, s, 0) V^T`` with rotations ``U``, ``V`` updated by Cayley steps, the seven-parameter factorization
    of Bartoli and Sturm, "Nonlinear estimation of the fundamental matrix with minimal parameters", TPAMI 2004. Each
    iteration takes the residuals and their Jacobian from two matrix products with per-correspondence monomials.
    ``x1`` and ``x2`` are homogeneous ``(N, 3)`` points, normalized by the caller. ``loss`` is ``"truncated"`` or
    ``"cauchy"`` with squared scale ``scale2``; ``mask`` (``(K, N)``) restricts each model to its correspondences. A
    step is kept only if it lowers the cost. For RANSAC, under ``torch.no_grad``. On CPU with gradients disabled, a
    singleton boolean mask compacts its correspondences, and the last trial evaluates only the cost: its Jacobian and
    updated optimizer state would not be used. Earlier iterations keep fused residual and Jacobian evaluation for small
    model batches.
    """
    K = F.shape[0]
    dtype, device = F.dtype, F.device
    cpu = K > 0 and device.type == "cpu" and not torch.is_grad_enabled()
    if cpu and K == 1 and mask is not None and mask.dtype == torch.bool:
        x1, x2, mask = x1[mask[0]], x2[mask[0]], None
    E = _hat_basis(dtype, device)
    eye3 = torch.eye(3, dtype=dtype, device=device)
    eye7 = torch.eye(7, dtype=dtype, device=device)
    algebraic = _epipolar_design_rows(x1, x2).T  # (9, N)
    quadratic = torch.cat([_epipolar_design_rows(x1, x1), _epipolar_design_rows(x2, x2)], 1).T  # (18, N)
    U, S, Vh = torch.linalg.svd(F)
    V = Vh.mT
    # Proper rotations: the third singular vectors do not enter F, so their signs are free.
    U = torch.cat([U[..., :2], U[..., 2:] * torch.linalg.det(U).sign()[:, None, None]], -1)
    V = torch.cat([V[..., :2], V[..., 2:] * torch.linalg.det(V).sign()[:, None, None]], -1)
    UV = torch.stack([U, V], 1)
    sigma = S[:, 1] / S[:, 0].clamp(min=torch.finfo(dtype).tiny)
    damping = torch.full((K, 1, 1), 1e-3, dtype=dtype, device=device)

    def compose(UV: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        scale = torch.stack([torch.ones_like(sigma), sigma], 1)[:, None, :]
        return (UV[:, 0, :, :2] * scale) @ UV[:, 1, :, :2].mT

    def normal_equations(F: torch.Tensor, UV: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Directions dF/dp: E_a F (left rotation), -F E_a (right rotation), u_2 v_2^T (singular value ratio).
        tangent = torch.cat(
            [E @ F[:, None], (F[:, None] @ E).neg(), (UV[:, 0, :, 1:2] @ UV[:, 1, :, 1:2].mT)[:, None]], 1
        )
        return _sampson_normal_equations(F, tangent, algebraic, quadratic, mask, loss, scale2)

    F = compose(UV, sigma)
    system, cost = normal_equations(F, UV)
    for iteration in range(iters):
        delta = -torch.linalg.solve_ex(system[..., :7] + damping * eye7, system[..., 7:])[0][..., 0]
        half = delta[:, :6].reshape(K * 2, 3) * 0.5
        skew = (half @ E.reshape(3, 9)).reshape(K, 2, 3, 3)
        factor = (2.0 / (1.0 + half.square().sum(1))).reshape(K, 2, 1, 1)
        UV_new = (eye3 + factor * (skew + skew @ skew)) @ UV  # Cayley transform of the half-angle skew matrix
        sigma_new = sigma + delta[:, 6]
        F_new = compose(UV_new, sigma_new)
        if cpu and iteration + 1 == iters:
            cost_new = _sampson_cost(F_new, algebraic, quadratic, mask, loss, scale2)
            accepted = cost_new < cost
            return torch.where(accepted[:, None, None], F_new, F)
        system_new, cost_new = normal_equations(F_new, UV_new)
        accept = (cost_new < cost)[:, None, None]
        UV = torch.where(accept[..., None], UV_new, UV)
        F, system = torch.where(accept, F_new, F), torch.where(accept, system_new, system)
        sigma, cost = torch.where(accept[:, 0, 0], sigma_new, sigma), torch.where(accept[:, 0, 0], cost_new, cost)
        damping = damping * torch.where(accept, 0.1, 10.0)
    return F


def run_7point(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    r"""Compute the fundamental matrix using the 7-point algorithm.

    The 7-point algorithm computes the fundamental matrix from exactly 7 point correspondences.
    Unlike the 8-point algorithm, this method returns 3 candidate fundamental matrices, one per real root of the
    cubic that formulates the rank-2 constraint, padded to three with zero matrices.

    Reference: Hartley/Zisserman 11.1.2 pag.281

    Args:
        points1: A set of 7 points in the first image with shape :math:`(B, 7, 2)`.
        points2: A set of 7 points in the second image with shape :math:`(B, 7, 2)`.

    Returns:
        The computed fundamental matrices with shape :math:`(B, 3, 3, 3)`, always 3 candidates per batch
        element. A cubic with a single real root gives one candidate followed by two zero matrices.

    """
    KORNIA_CHECK_SHAPE(points1, ["B", "7", "2"])
    KORNIA_CHECK_SHAPE(points2, ["B", "7", "2"])

    # Reference: Hartley and Zisserman, section 11.1.2; the cubic follows OpenCV's run7Point
    # (https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/fundam.cpp).
    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)
    A = _epipolar_design_rows(points1_norm, points2_norm)
    candidates, valid = _seven_point_candidates(A)
    # F = T2^T F T1
    fmatrix = normalize_transformation(transform2[:, None].mT @ candidates @ transform1[:, None])
    return torch.where(valid[..., None, None], fmatrix, torch.zeros_like(fmatrix))


def run_8point(
    points1: torch.Tensor,
    points2: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    use_einsum_at_more_than_points: int = 512,
) -> torch.Tensor:
    r"""Compute the fundamental matrix using (weighted) 8-point DLT, optimized.

    Args:
        points1: (B, N, 2), N >= 8
        points2: (B, N, 2), N >= 8
        weights: optional (B, N) nonnegative weights
        use_einsum_at_more_than_points: threshold for using einsum vs GEMM for large N

    Returns:
        (B, 3, 3) fundamental matrices
    """
    KORNIA_CHECK_SHAPE(points1, ["B", "N", "2"])
    KORNIA_CHECK_SHAPE(points2, ["B", "N", "2"])
    KORNIA_CHECK_SAME_SHAPE(points1, points2)
    if points1.shape[1] < 8:
        raise AssertionError(points1.shape)
    if weights is not None:
        KORNIA_CHECK_SHAPE(weights, ["B", "N"])
        if weights.shape[1] != points1.shape[1]:
            raise AssertionError(weights.shape)

    # Use the same correspondences for Hartley statistics and the weighted DLT system.
    pts1n, T1 = normalize_points(points1, weights=weights)
    pts2n, T2 = normalize_points(points2, weights=weights)

    # Design matrix rows A_i = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    # Shape: A ∈ (B, N, 9)
    A = _epipolar_design_rows(pts1n, pts2n)

    B, N, _ = A.shape

    if weights is None and N == 8:
        # A minimal sample has an exact null vector.
        h = _eight_point_null_vector(A)
    else:
        # Build normal matrix M = A^T W A  (B,9,9) without forming NxN diagonals.
        if weights is None:
            if N < use_einsum_at_more_than_points:
                # Use GEMM on tall A: (B,9,N) @ (B,N,9)
                M = A.transpose(-2, -1).contiguous() @ A
            else:
                # Accumulate via einsum (saves bandwidth for huge N)
                M = torch.einsum("bni,bnj->bij", A, A)
        else:
            # Negative weights count as zero. A weight of exactly 0 is the documented way to drop a correspondence,
            # and its gradient should be the one-sided derivative from above. ``clamp_min(0)`` passes the gradient
            # through at the bound on torch 2.5.1 and 2.9.1 but returns 0 on 2.14 (#4229); ``where`` passes it on
            # every version.
            w = torch.where(weights < 0, 0.0, weights)
            if N < use_einsum_at_more_than_points:
                # Scale one factor by w instead of both by sqrt(w). Both build the same A^T W A, but the
                # derivative of sqrt is unbounded at 0, so a zero weight got a NaN gradient. This form is linear in
                # w, like the einsum branch below.
                Aw = A * w.unsqueeze(-1)
                M = Aw.transpose(-2, -1).contiguous() @ A
            else:
                # Weighted einsum
                M = torch.einsum("bni,bnj,bn->bij", A, A, w)

        _evals, evecs = torch.linalg.eigh(M)  # ascending order
        h = evecs[..., 0]  # (B,9), eigenvector for smallest λ
    F_rank2 = _enforce_rank2(h.reshape(B, 3, 3))
    F = T2.transpose(-2, -1) @ (F_rank2 @ T1)

    return normalize_transformation(F)


def find_fundamental(
    points1: torch.Tensor,
    points2: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    method: Literal["8POINT", "7POINT"] = "8POINT",
) -> torch.Tensor:
    r"""Find the fundamental matrix.

    Convention:
        - ``points1`` are in the first image and ``points2`` in the second; the result satisfies
          :math:`x_2^\top F x_1 = 0`, the second image's point on the left.
          :ref:`Two-view geometry <two-view-conventions>` maps this onto OpenCV.
        - The result is scaled so that ``F[2, 2] = 1`` by :func:`normalize_transformation`, which leaves it at its
          unnormalised scale when ``F[2, 2]`` is numerically zero, as for exactly rectified stereo.
          ``method="7POINT"`` returns three candidates in no particular order.
        - ``weights`` weight each correspondence's equation in the linear system: only their ratios matter, a
          negative weight counts as zero, and ``method="7POINT"`` ignores them.
        - When the 7-point cubic has one real root, the two extra ``"7POINT"`` candidates are zero matrices.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`: :math:`N \ge 8` for
            ``"8POINT"``, exactly 7 for ``"7POINT"``.
        points2: A set of points in the second image with the shape of ``points1``.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
        method: The method to use for computing the fundamental matrix. Supported methods are "7POINT" and "8POINT".

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3, 3)` for ``"8POINT"`` and
        :math:`(B, 3, 3, 3)` for ``"7POINT"``.

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


def compute_correspond_epilines(points: torch.Tensor, F_mat: torch.Tensor) -> torch.Tensor:
    r"""Compute the corresponding epipolar line for a given set of points.

    Convention:
        - For first-image points and ``F`` from :func:`find_fundamental` the lines lie in the second image;
          for second-image points pass ``F.transpose(-2, -1)``. Lines are scaled to :math:`a^2 + b^2 = 1`.

    Args:
        points: tensor containing the set of points to project in the shape of :math:`(*, N, 2)` or :math:`(*, N, 3)`.
        F_mat: the fundamental to use for projection the points in the shape of :math:`(*, 3, 3)`.

    Returns:
        a tensor with shape :math:`(*, N, 3)` containing the epipolar lines :math:`F x` of the points.
        Each line is described as
        :math:`ax + by + c = 0` and encoding the vectors as :math:`(a, b, c)`.

    """
    KORNIA_CHECK_SHAPE(points, ["*", "N", "DIM"])
    if points.shape[-1] == 2:
        points_h: torch.Tensor = convert_points_to_homogeneous(points)
    elif points.shape[-1] == 3:
        points_h = points
    else:
        raise AssertionError(points.shape)
    KORNIA_CHECK_SHAPE(F_mat, ["*", "3", "3"])
    # project points and retrieve lines components
    points_h = torch.transpose(points_h, dim0=-2, dim1=-1)
    a, b, c = torch.chunk(F_mat @ points_h, dim=-2, chunks=3)

    # compute normal and compose equation line
    nu: torch.Tensor = a * a + b * b
    nu = torch.where(nu > 0.0, 1.0 / torch.sqrt(nu), torch.ones_like(nu))

    line = torch.cat([a * nu, b * nu, c * nu], dim=-2)  # *x3xN
    return torch.transpose(line, dim0=-2, dim1=-1)  # *xNx3


def get_perpendicular(lines: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    r"""Compute the perpendicular to a line, through the point.

    Args:
        lines: tensor containing the set of lines :math:`(B, N, 3)`.
        points:  tensor containing the set of points :math:`(B, N, 2)`.

    Returns:
        a tensor with shape :math:`(B, N, 3)` containing a vector of the epipolar
        perpendicular lines. Each line is described as
        :math:`ax + by + c = 0` and encoding the vectors as :math:`(a, b, c)`; the normal :math:`(a, b)` has the
        norm of the input line's and is not rescaled.

    """
    KORNIA_CHECK_SHAPE(lines, ["*", "N", "3"])
    KORNIA_CHECK_SHAPE(points, ["*", "N", "two"])
    if points.shape[2] == 2:
        points_h: torch.Tensor = convert_points_to_homogeneous(points)
    elif points.shape[2] == 3:
        points_h = points
    else:
        raise AssertionError(points.shape)
    infinity_point = lines * torch.tensor([1, 1, 0], dtype=lines.dtype, device=lines.device).view(1, 1, 3)
    perp: torch.Tensor = torch.linalg.cross(points_h, infinity_point, dim=2)
    return perp


def get_closest_point_on_epipolar_line(pts1: torch.Tensor, pts2: torch.Tensor, Fm: torch.Tensor) -> torch.Tensor:
    r"""Return closest point on the epipolar line to the correspondence, given the fundamental matrix.

    Convention:
        - Returns, in the second image, the point of the epipolar line of ``pts1`` closest to ``pts2``, for
          ``Fm`` in the :math:`x_2^\top F x_1 = 0` order of :func:`find_fundamental`.

    Args:
        pts1: points in the first image with shape :math:`(B, N, 2)` or :math:`(B, N, 3)`. If they are not
              homogeneous, converted automatically.
        pts2: points in the second image with shape :math:`(B, N, 2)` or :math:`(B, N, 3)`. If they are not
              homogeneous, converted automatically.
        Fm: Fundamental matrices with shape :math:`(B, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.

    Returns:
        point on epipolar line :math:`(B, N, 2)`.

    """
    if not isinstance(Fm, torch.Tensor):
        raise TypeError(f"Fm type is not a torch.Tensor. Got {type(Fm)}")
    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")
    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)
    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)
    line1in2 = compute_correspond_epilines(pts1, Fm)
    perp = get_perpendicular(line1in2, pts2)
    return convert_points_from_homogeneous(torch.linalg.cross(line1in2, perp, dim=2))


def fundamental_from_essential(E_mat: torch.Tensor, K1: torch.Tensor, K2: torch.Tensor) -> torch.Tensor:
    r"""Get the Fundamental matrix from Essential and camera matrices.

    Uses the method from Hartley/Zisserman 9.6 pag 257 (formula 9.12).

    Convention:
        - :math:`F = K_2^{-\top} E K_1^{-1}` with ``K1`` the camera of the first image, so ``F`` follows the
          :math:`x_2^\top F x_1 = 0` order of :func:`find_fundamental`; it keeps the scale of ``E_mat``, with no
          ``F[2, 2] = 1`` normalisation.

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


def fundamental_from_projections(P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
    r"""Get the Fundamental matrix from Projection matrices.

    Convention:
        - The result satisfies :math:`x_2^\top F x_1 = 0` for ``(P1, P2)``, the order of :func:`find_fundamental`.
          It is not normalised, and for ``P1 = [I | 0]``, ``P2 = [R | t]`` it is the negative of
          :func:`~kornia.geometry.epipolar.essential_from_Rt` for the same motion.
        - float16 and bfloat16 inputs are computed in float32. A float16 ``F`` is then divided by its largest
          absolute entry before the cast back, because pixel-unit projection matrices give entries of order ``1e10``,
          above float16's maximum of 65504; an all-zero ``F`` (coincident camera centres) stays zero. bfloat16 has
          float32's exponent range and is cast back unnormalised.

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

    def vstack(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, y], dim=-2)

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

    F = F_vec.view(*P1.shape[:-2], 3, 3)

    if input_dtype == torch.float16:
        # F is defined up to scale: divide by the largest absolute entry so it fits float16; F = 0 stays 0.
        scale = F.abs().amax(dim=(-2, -1), keepdim=True)
        F = F / torch.where(scale > 0, scale, torch.ones_like(scale))

    return F.to(input_dtype)
