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

"""Batched kernels of RANSAC's Levenberg-Marquardt pipeline for fundamental matrices and homographies.

The minimal solvers work on correspondences normalized once per call and take their null spaces from a
partial-pivoted LU factorization, which is batched on every backend (``eigh`` and ``svd`` loop over the batch on
CPU, and the batched CUDA SVD is slow for 3x3 matrices). Residuals are scored from one matrix product with
per-correspondence monomials. The refiners are Levenberg-Marquardt iterations batched over models, in the spirit of
PoseLib's (Larsson and contributors, https://github.com/PoseLib/PoseLib) ``refine_fundamental`` and
``refine_homography``: fundamental matrices are parametrized by the SVD-based factorization of Bartoli and Sturm,
"Nonlinear estimation of the fundamental matrix with minimal parameters", TPAMI 2004, and homographies by a
tangent step of the unit-norm matrix.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

__all__: list[str] = []


def _solve_dtype(device: torch.device) -> torch.dtype:
    """The dtype of the small closed-form steps: float64, except on MPS, which has none."""
    return torch.float32 if device.type == "mps" else torch.float64


def normalize_correspondences(
    kp1: torch.Tensor, kp2: torch.Tensor, shared_scale: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    r"""Center each image's points and scale them so the mean distance to the centroid is :math:`\sqrt{2}`.

    Statistics use the correspondences that are finite in both images; the others stay non-finite and so are never
    counted as inliers. ``shared_scale`` uses one scale for both images, which keeps the Sampson distance a multiple
    of the pixel one.

    Returns:
        Homogeneous normalized points ``(N, 3)`` of each image, the ``(3, 3)`` transforms that map pixels to them, and
        the two scales (pixels per normalized unit).

    """
    finite = torch.isfinite(kp1).all(1) & torch.isfinite(kp2).all(1)
    count = finite.sum().clamp(min=1).to(kp1.dtype)
    weight = finite.to(kp1.dtype)[:, None]
    zero = torch.zeros_like(kp1)
    p1 = torch.where(finite[:, None], kp1, zero)
    p2 = torch.where(finite[:, None], kp2, zero)
    c1 = (p1 * weight).sum(0) / count
    c2 = (p2 * weight).sum(0) / count
    r1 = ((p1 - c1).norm(dim=1) * weight[:, 0]).sum() / count
    r2 = ((p2 - c2).norm(dim=1) * weight[:, 0]).sum() / count
    if shared_scale:
        r1 = r2 = (r1 + r2) / 2
    s1 = float(r1) / math.sqrt(2.0)
    s2 = float(r2) / math.sqrt(2.0)
    # Coincident points leave no scale; any positive one keeps the transform invertible.
    s1 = s1 if math.isfinite(s1) and s1 > 0 else 1.0
    s2 = s2 if math.isfinite(s2) and s2 > 0 else 1.0
    ones = torch.ones_like(kp1[:, :1])
    x1 = torch.cat([(kp1 - c1) / s1, ones], 1)
    x2 = torch.cat([(kp2 - c2) / s2, ones], 1)
    t1 = torch.eye(3, dtype=kp1.dtype, device=kp1.device)
    t2 = torch.eye(3, dtype=kp1.dtype, device=kp1.device)
    t1[0, 0] = t1[1, 1] = 1.0 / s1
    t2[0, 0] = t2[1, 1] = 1.0 / s2
    t1[:2, 2] = -c1 / s1
    t2[:2, 2] = -c2 / s2
    return x1, x2, t1, t2, s1, s2


def null_space_lu(A: torch.Tensor) -> torch.Tensor:
    """Right null spaces of a batch of full-row-rank ``(B, m, n)`` matrices, ``m < n``, as ``(B, n, n - m)``.

    With ``A^T = P L U`` from a partial-pivoted LU factorization, ``f^T A^T = 0`` exactly when ``y = P^T f`` solves
    ``y^T L = 0``. Splitting the unit lower trapezoidal ``L`` into its square top ``L_1`` and bottom ``L_2`` rows
    gives the basis ``y = [-(L_2 L_1^{-1})^T; I]``. The pivoting chooses the gauge, so no coordinate of the null
    vector is assumed non-zero. A rank-deficient ``A`` gives non-finite vectors, which the callers discard.
    """
    batch, m, n = A.shape
    lu, pivots, _ = torch.linalg.lu_factor_ex(A.mT)
    lower = torch.linalg.solve_triangular(lu[:, :m, :m], lu[:, m:, :m], upper=False, left=False, unitriangular=True)
    eye = torch.eye(n - m, dtype=A.dtype, device=A.device).expand(batch, -1, -1)
    permutation, _, _ = torch.lu_unpack(lu, pivots, unpack_data=False)
    return permutation @ torch.cat([-lower.mT, eye], 1)


def _det3(M: torch.Tensor) -> torch.Tensor:
    return (M[..., 0, :] * torch.linalg.cross(M[..., 1, :], M[..., 2, :])).sum(-1)


def rank2_projection(F: torch.Tensor) -> torch.Tensor:
    r"""The nearest rank-2 matrices in Frobenius norm, ``F (I - v v^T)`` with ``v`` the smallest right singular vector.

    ``v`` is the eigenvector of ``F^T F`` for its smallest eigenvalue, from the trigonometric solution of the 3x3
    characteristic polynomial and a cross product of two rows of ``F^T F - \lambda I``; unlike a batched SVD this is
    a few elementwise operations on every backend.
    """
    M = F.mT @ F
    q = M.diagonal(dim1=-2, dim2=-1).sum(-1) / 3
    eye = torch.eye(3, dtype=F.dtype, device=F.device)
    shifted = M - q[:, None, None] * eye
    p = (shifted.square().sum((-2, -1)) / 6).sqrt()
    safe_p = torch.where(p > 0, p, torch.ones_like(p))
    r = (_det3(shifted / safe_p[:, None, None]) / 2).clamp(-1, 1)
    smallest = q + 2 * p * torch.cos(torch.acos(r) / 3 + 2 * math.pi / 3)
    rows = M - smallest[:, None, None] * eye
    crosses = torch.linalg.cross(rows[:, [0, 0, 1]], rows[:, [1, 2, 2]])
    norms = crosses.square().sum(-1)
    best = norms.argmax(1)
    v = crosses.gather(1, best[:, None, None].expand(-1, 1, 3))[:, 0]
    v = v * norms.gather(1, best[:, None]).clamp(min=torch.finfo(F.dtype).tiny).rsqrt()
    return F - (F @ v[:, :, None]) @ v[:, None, :]


def _epipolar_rows(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Rows ``vec(x2 x1^T)`` of the epipolar constraint, so that ``row . vec(F) = x2^T F x1`` (``F`` row-major)."""
    return (x2[..., :, None] * x1[..., None, :]).flatten(-2)


def fundamental_8pt(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Rank-2 fundamental matrices ``(B, 3, 3)`` from eight homogeneous normalized correspondences ``(B, 8, 3)``."""
    f = null_space_lu(_epipolar_rows(x1, x2))[..., 0]
    F = (f * f.square().sum(1, keepdim=True).rsqrt()).reshape(-1, 3, 3)
    return rank2_projection(F.to(_solve_dtype(F.device))).to(x1.dtype)


def _cubic_real_roots(c3: torch.Tensor, c2: torch.Tensor, c1: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
    """Real roots ``(B, 3)`` of ``c3 x^3 + c2 x^2 + c1 x + c0``, NaN where a cubic has only one.

    Cardano's formula for one real root, the trigonometric one for three, each followed by a Newton step. The caller
    arranges ``|c3| >= |c0|`` so that the leading coefficient is not the vanishing one.
    """
    a, b, c = c2 / c3, c1 / c3, c0 / c3
    a3 = a / 3
    p = b - a * a3
    q = (2 * a3 * a3 - b) * a3 + c
    discriminant = 0.25 * q * q + p * p * p / 27
    three = discriminant <= 0
    root = discriminant.clamp(min=0).sqrt()
    u, w = root - 0.5 * q, -root - 0.5 * q
    single = torch.copysign(u.abs().pow(1 / 3), u) + torch.copysign(w.abs().pow(1 / 3), w)
    radius = (-p / 3).clamp(min=0).sqrt()
    safe_radius = torch.where(radius > 0, radius, torch.ones_like(radius))
    angle = torch.acos((-0.5 * q / safe_radius.pow(3)).clamp(-1, 1)) / 3
    offsets = torch.tensor([0.0, 2 * math.pi / 3, 4 * math.pi / 3], dtype=c3.dtype, device=c3.device)
    triple = 2 * radius[:, None] * torch.cos(angle[:, None] - offsets)
    x = torch.where(three[:, None], triple, single[:, None].expand(-1, 3)) - a3[:, None]
    value = ((x + a[:, None]) * x + b[:, None]) * x + c[:, None]
    slope = (3 * x + 2 * a[:, None]) * x + b[:, None]
    x = x - value / torch.where(slope == 0, torch.ones_like(slope), slope)
    only_one = torch.stack([torch.zeros_like(three), ~three, ~three], 1)
    return x.masked_fill(only_one, float("nan"))


def fundamental_7pt(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Up to three fundamental matrices ``(B, 3, 3, 3)`` per seven correspondences ``(B, 7, 3)``; NaN where fewer.

    The two-dimensional null space ``F(x) = x f_1 + f_2`` of the epipolar constraints is completed by the real roots
    of the cubic ``det F(x) = 0`` (Hartley and Zisserman, section 11.1.2), so every candidate has rank two.
    """
    basis = null_space_lu(_epipolar_rows(x1, x2)).mT.reshape(-1, 2, 3, 3).to(_solve_dtype(x1.device))
    f1, f2 = basis[:, 0], basis[:, 1]
    # det(x f1 + f2) by multilinearity in the rows: c3 = det f1, c0 = det f2.
    a2, a3, b2, b3 = f1[:, 1], f1[:, 2], f2[:, 1], f2[:, 2]
    crosses = torch.linalg.cross(torch.stack([a2, b2, a2, b2], 1), torch.stack([a3, b3, b3, a3], 1))
    x_aa, x_bb, x_ab = crosses[:, 0], crosses[:, 1], crosses[:, 2] + crosses[:, 3]
    a1, b1 = f1[:, 0], f2[:, 0]
    dots = (torch.stack([a1, b1, a1, a1, b1, b1], 1) * torch.stack([x_aa, x_aa, x_ab, x_bb, x_ab, x_bb], 1)).sum(-1)
    coefficients = torch.stack([dots[:, 0], dots[:, 1] + dots[:, 2], dots[:, 3] + dots[:, 4], dots[:, 5]], 1)
    # Parametrize by the better-conditioned end: F = x f1 + f2, or F = f1 + y f2 with the coefficients reversed.
    swap = coefficients[:, 0].abs() < coefficients[:, 3].abs()
    coefficients = torch.where(swap[:, None], coefficients.flip(1), coefficients)
    roots = _cubic_real_roots(*coefficients.unbind(1))
    lead = torch.where(swap[:, None, None], f2, f1)
    rest = torch.where(swap[:, None, None], f1, f2)
    F = roots[:, :, None, None] * lead[:, None] + rest[:, None]
    return (F * F.square().sum((-2, -1), keepdim=True).rsqrt()).to(x1.dtype)


def homography_4pt(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Homographies ``(B, 3, 3)``, of unit Frobenius norm, from four normalized correspondences ``(B, 4, 3)``."""
    zero = torch.zeros_like(x1)
    rows_u = torch.cat([x1, zero, -x2[..., 0:1] * x1], -1)
    rows_v = torch.cat([zero, x1, -x2[..., 1:2] * x1], -1)
    h = null_space_lu(torch.stack([rows_u, rows_v], 2).flatten(1, 2))[..., 0]
    return (h * h.square().sum(1, keepdim=True).rsqrt()).reshape(-1, 3, 3)


def sampson_basis(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Per-correspondence monomials ``(27, 2N)`` for the residuals and gradient norms of the Sampson distance.

    ``[vec F, vec Q1, vec Q2] @ basis`` gives the epipolar residuals and the squared gradient norms, with
    ``Q1 = F[:2]^T F[:2]`` and ``Q2 = F[:, :2] F[:, :2]^T``. Both halves are linear in these per-model quantities:
    ``x2^T F x1`` in ``vec(x2 x1^T)``, and ``|F[:2] x1|^2 + |F[:, :2]^T x2|^2 = x1^T Q1 x1 + x2^T Q2 x2`` in
    ``vec(x1 x1^T)`` and ``vec(x2 x2^T)``.
    """
    n = x1.shape[0]
    basis = x1.new_zeros(27, 2 * n)
    basis[:9, :n] = _epipolar_rows(x1, x2).T
    basis[9:18, n:] = _epipolar_rows(x1, x1).T
    basis[18:, n:] = _epipolar_rows(x2, x2).T
    return basis


def sampson_errors(F: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Squared Sampson distances ``(B, N)`` of fundamental matrices ``(B, 3, 3)`` from :func:`sampson_basis`."""
    n = basis.shape[1] // 2
    q1 = F[:, :2, :].mT @ F[:, :2, :]
    q2 = F[:, :, :2] @ F[:, :, :2].mT
    out = torch.cat([F.flatten(1), q1.flatten(1), q2.flatten(1)], 1) @ basis
    return out[:, :n].square() / out[:, n:]


def transfer_basis(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Per-correspondence monomials ``(9, 3N)`` for the one-way transfer error.

    ``vec(H) @ basis`` is ``[P_0 - u P_2 | P_1 - v P_2 | P_2]`` for ``P = H x1`` and ``x2 = (u, v)``.
    """
    n = x1.shape[0]
    basis = x1.new_zeros(9, 3 * n)
    basis[0:3, :n] = x1.T
    basis[6:9, :n] = -(x2[:, 0:1] * x1).T
    basis[3:6, n : 2 * n] = x1.T
    basis[6:9, n : 2 * n] = -(x2[:, 1:2] * x1).T
    basis[6:9, 2 * n :] = x1.T
    return basis


def transfer_errors(H: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Squared one-way transfer errors ``(B, N)`` of homographies ``(B, 3, 3)`` from :func:`transfer_basis`."""
    n = basis.shape[1] // 3
    out = H.flatten(1) @ basis
    return (out[:, :n].square() + out[:, n : 2 * n].square()) / out[:, 2 * n :].square()


def _robust(r2: torch.Tensor, loss: str, scale2: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """IRLS weights and costs of a squared residual: Cauchy, or truncated at ``scale2``."""
    if loss == "cauchy":
        return 1.0 / (1.0 + r2 / scale2), torch.log1p(r2 / scale2)
    return (r2 < scale2).to(r2.dtype), torch.fmin(r2, torch.full_like(r2[:1, :1], scale2))


def _hat_basis(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """``E[a] = [e_a]_x``, the generators of rotations, as ``(3, 3, 3)``."""
    E = torch.zeros(3, 3, 3, dtype=dtype, device=device)
    E[0, 1, 2], E[0, 2, 1] = -1.0, 1.0
    E[1, 0, 2], E[1, 2, 0] = 1.0, -1.0
    E[2, 0, 1], E[2, 1, 0] = -1.0, 1.0
    return E


def refine_fundamental(
    F: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor | None, loss: str, scale2: float, iters: int
) -> torch.Tensor:
    """Levenberg-Marquardt on the Sampson distance, batched over fundamental matrices ``(K, 3, 3)``.

    ``F = U diag(1, s, 0) V^T`` with rotations ``U``, ``V`` updated by Cayley steps, seven parameters (Bartoli and
    Sturm). Each iteration takes the residuals and their Jacobian from two matrix products with per-correspondence
    monomials, like :func:`sampson_errors`. ``loss`` is ``"truncated"`` or ``"cauchy"`` with squared scale ``scale2``;
    ``mask`` (``(K, N)``) restricts each model to its correspondences. A step is kept only if it lowers the cost.
    """
    K = F.shape[0]
    dtype, device = F.dtype, F.device
    E = _hat_basis(dtype, device)
    eye3 = torch.eye(3, dtype=dtype, device=device)
    eye7 = torch.eye(7, dtype=dtype, device=device)
    algebraic = _epipolar_rows(x1, x2).T  # (9, N)
    quadratic = torch.cat([_epipolar_rows(x1, x1), _epipolar_rows(x2, x2)], 1).T  # (18, N)
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
        stacked = torch.cat([F[:, None], tangent], 1)  # (K, 8, 3, 3): F, then the seven directions
        # x1^T (F[:2]^T X[:2]) x1 + x2^T (F[:, :2] X[:, :2]^T) x2 is half the derivative of the squared gradient norm.
        quad1 = F[:, None, :2, :].mT @ stacked[:, :, :2, :]
        quad2 = F[:, None, :, :2] @ stacked[:, :, :, :2].mT
        out_c = stacked.reshape(K, 8, 9) @ algebraic
        out_g = torch.cat([quad1, quad2], 2).reshape(K, 8, 18) @ quadratic
        inv = out_g[:, 0].rsqrt()
        r = out_c[:, 0] * inv
        J = (out_c[:, 1:] - (r * inv)[:, None] * out_g[:, 1:]) * inv[:, None]  # (K, 7, N)
        w, rho = _robust(r * r, loss, scale2)
        if mask is not None:
            w, rho = w * mask, rho * mask
        Jw = J * w[:, None]
        return torch.cat([Jw @ J.mT, Jw @ r[..., None]], 2), rho.sum(1)

    F = compose(UV, sigma)
    system, cost = normal_equations(F, UV)
    for _ in range(iters):
        delta = -torch.linalg.solve_ex(system[..., :7] + damping * eye7, system[..., 7:])[0][..., 0]
        half = delta[:, :6].reshape(K * 2, 3) * 0.5
        skew = (half @ E.reshape(3, 9)).reshape(K, 2, 3, 3)
        factor = (2.0 / (1.0 + half.square().sum(1))).reshape(K, 2, 1, 1)
        UV_new = (eye3 + factor * (skew + skew @ skew)) @ UV  # Cayley transform of the half-angle skew matrix
        sigma_new = sigma + delta[:, 6]
        F_new = compose(UV_new, sigma_new)
        system_new, cost_new = normal_equations(F_new, UV_new)
        accept = (cost_new < cost)[:, None, None]
        UV = torch.where(accept[..., None], UV_new, UV)
        F, system = torch.where(accept, F_new, F), torch.where(accept, system_new, system)
        sigma, cost = torch.where(accept[:, 0, 0], sigma_new, sigma), torch.where(accept[:, 0, 0], cost_new, cost)
        damping = damping * torch.where(accept, 0.1, 10.0)
    return F


def refine_homography(
    H: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor | None, loss: str, scale2: float, iters: int
) -> torch.Tensor:
    """Levenberg-Marquardt on the one-way transfer error, batched over homographies ``(K, 3, 3)``.

    ``x1`` is homogeneous ``(N, 3)`` and ``x2`` ``(N, 2)``. Steps are taken in the eight-dimensional orthogonal
    complement of the unit-norm ``vec(H)`` and renormalized; the loss, ``mask`` and step acceptance are those of
    :func:`refine_fundamental`.
    """
    K = H.shape[0]
    dtype, device = H.dtype, H.device
    eye8 = torch.eye(8, dtype=dtype, device=device)
    eye9 = torch.eye(9, dtype=dtype, device=device)
    last = eye9[8]
    target = torch.cat([x2[:, 0], x2[:, 1]])
    h = H.flatten(1)
    h = h * h.square().sum(1, keepdim=True).rsqrt()
    damping = torch.full((K, 1, 1), 1e-3, dtype=dtype, device=device)

    def normal_equations(h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Householder reflection of h onto the last axis: its other eight columns span the tangent space.
        v = h + torch.where(h[:, 8:9] >= 0, 1.0, -1.0) * last
        v = v * v.square().sum(1, keepdim=True).rsqrt()
        tangent = (eye9 - 2 * v[:, :, None] * v[:, None, :])[:, :, :8]  # (K, 9, 8)
        stacked = torch.cat([h[:, :, None], tangent], 2).mT.reshape(K, 27, 3)  # rows of H, then of each direction
        P = (stacked @ x1.T).reshape(K, 9, 3, -1)  # (K, 9, 3, N): P = H x1, then its directional derivatives
        iz = 1.0 / P[:, 0, 2]
        uv = P[:, 0, :2] * iz[:, None]  # (K, 2, N)
        r = uv.flatten(1) - target  # (K, 2N): u residuals, then v residuals
        J = ((P[:, 1:, :2] - uv[:, None] * P[:, 1:, 2:3]) * iz[:, None, None]).flatten(2)  # (K, 8, 2N)
        r2 = r[:, : x1.shape[0]].square() + r[:, x1.shape[0] :].square()
        w, rho = _robust(r2, loss, scale2)
        if mask is not None:
            w, rho = w * mask, rho * mask
        Jw = J * torch.cat([w, w], 1)[:, None]
        return torch.cat([Jw @ J.mT, Jw @ r[..., None]], 2), rho.sum(1), tangent

    system, cost, tangent = normal_equations(h)
    for _ in range(iters):
        delta = -torch.linalg.solve_ex(system[..., :8] + damping * eye8, system[..., 8:])[0]
        h_new = h + (tangent @ delta)[..., 0]
        h_new = h_new * h_new.square().sum(1, keepdim=True).rsqrt()
        system_new, cost_new, tangent_new = normal_equations(h_new)
        accept = (cost_new < cost)[:, None, None]
        h, cost = torch.where(accept[:, :, 0], h_new, h), torch.where(accept[:, 0, 0], cost_new, cost)
        system, tangent = torch.where(accept, system_new, system), torch.where(accept, tangent_new, tangent)
        damping = damping * torch.where(accept, 0.1, 10.0)
    return h.reshape(K, 3, 3)
