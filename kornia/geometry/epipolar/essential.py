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

"""Module containing functionalities for the Essential matrix."""

from typing import Any, Optional, Tuple, Union

import torch

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SAME_SHAPE, KORNIA_CHECK_SHAPE
from kornia.core.ops import eye_like, vec_like
from kornia.core.utils import _torch_svd_cast
from kornia.geometry.solvers.homogeneous import _null_space_householder
from kornia.geometry.solvers.polynomial_solver import T_deg1, T_deg2

from .fundamental import _epipolar_design_rows, _hat_basis, _sampson_cost, _sampson_normal_equations, _solve_dtype
from .numeric import cross_product_matrix, matrix_cofactor_tensor
from .projection import depth_from_point, projection_from_KRt
from .triangulation import triangulate_points

__all__ = [
    "decompose_essential_matrix",
    "decompose_essential_matrix_no_svd",
    "essential_from_Rt",
    "essential_from_fundamental",
    "find_essential",
    "motion_from_essential",
    "motion_from_essential_choose_solution",
    "project_to_essential",
    "relative_camera_motion",
]


def run_5point(points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Compute the essential matrix using the 5-point algorithm from Nister.

    The linear system is solved by Nister's 5-point algorithm [@nister2004efficient],
    and the solver implemented referred to [@barath2020magsac++][@wei2023generalized][@wang2023vggsfm].

    Args:
        points1: A set of calibrated points in the first image with a tensor shape :math:`(B, N, 2), N>=5`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=5`.
        weights: Not used, kept for compatibility.

    Returns:
        the computed essential matrix with shape :math:`(B, 10, 3, 3)`.

    """
    KORNIA_CHECK_SHAPE(points1, ["B", "N", "2"])
    KORNIA_CHECK_SAME_SHAPE(points1, points2)
    KORNIA_CHECK(points1.shape[1] >= 5, "Number of points should be >=5")
    # Rows vec(x2 x1^T), so that a null vector reshapes row-major to E.
    design = _epipolar_design_rows(points1, points2)
    # A sample without a real root keeps ten NaN slots, like the complex slots of any other sample.
    if design.shape[1] == 5:
        candidates, _ = _five_point_candidates(design)
    else:
        candidates = _least_squares_candidates(design)
    return candidates


def _polynomial_product_table(m: int, n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """``T[i, j, i + j] = 1``: the coefficients of a product of polynomials with ``m`` and ``n`` coefficients."""
    i, j = torch.meshgrid(torch.arange(m, device=device), torch.arange(n, device=device), indexing="ij")
    table = torch.zeros(m, n, m + n - 1, dtype=dtype, device=device)
    table[i, j, i + j] = 1.0
    return table


def _determinant_to_polynomial_jit(A: torch.Tensor) -> torch.Tensor:
    """Coefficients ``(B, 11)``, of ``z^0`` first, of the determinant of Nister's hidden-variable matrix ``(B, 3, 13)``.

    Each row of ``A`` holds three polynomials in ``z``, highest degree first: two cubics (columns 0-3 and 4-7) and a
    quartic (8-12). The determinant is the sum over the six permutations of the rows of signed products of one
    polynomial of each kind, expanded with polynomial product tables: a few matrix products, deterministic on every
    backend, where the explicit expansion into 486 triple products cost five times as much.
    """
    device, dtype = A.device, A.dtype
    levi_civita = torch.zeros(3, 3, 3, dtype=dtype, device=device)
    for (a, b, c), sign in (
        ((0, 1, 2), 1.0),
        ((1, 2, 0), 1.0),
        ((2, 0, 1), 1.0),
        ((0, 2, 1), -1.0),
        ((2, 1, 0), -1.0),
        ((1, 0, 2), -1.0),
    ):
        levi_civita[a, b, c] = sign
    cubic1, cubic2, quartic = A[..., 0:4].flip(-1), A[..., 4:8].flip(-1), A[..., 8:13].flip(-1)
    sextics = torch.einsum("xai,xbj,ijm->xabm", cubic1, cubic2, _polynomial_product_table(4, 4, dtype, device))
    signed = torch.einsum("xabm,abc->xcm", sextics, levi_civita)
    return torch.einsum("xcm,xck,mkn->xn", signed, quartic, _polynomial_product_table(7, 5, dtype, device))


def _solve_2x2_tikhonov_safe(A: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Solve (A)x=b for A (...,2,2), b (...,2,1) using Tikhonov regularization.

    Uses the following methods:
      - direct inverse when det is OK
      - otherwise solve normal equations (A^T A + λI)x = A^T b  (λ from trace scale)
    Never throws. Returns (x, bad) where bad marks ill-conditioned A.
    """
    a = A[..., 0, 0]
    bb = A[..., 0, 1]
    c = A[..., 1, 0]
    d = A[..., 1, 1]

    det = a * d - bb * c
    det_abs = det.abs()
    bad = (det_abs <= eps) | torch.isnan(det_abs) | torch.isinf(det_abs)

    # ---- direct inverse branch (but branchless via where) ----
    det_safe = torch.where(det_abs > eps, det, torch.ones_like(det) * eps)
    inv_det = 1.0 / det_safe

    inv00 = d * inv_det
    inv01 = (-bb) * inv_det
    inv10 = (-c) * inv_det
    inv11 = a * inv_det

    x0_dir = inv00 * b[..., 0, 0] + inv01 * b[..., 1, 0]
    x1_dir = inv10 * b[..., 0, 0] + inv11 * b[..., 1, 0]
    x_dir = torch.stack((x0_dir, x1_dir), dim=-1).unsqueeze(-1)  # (...,2,1)

    # ---- fallback: normal equations with λI (always SPD if λ>0) ----
    # ATA = A^T A, ATb = A^T b
    # ATA = [[a^2 + c^2, a*bb + c*d],
    #        [a*bb + c*d, bb^2 + d^2]]
    ata00 = a * a + c * c
    ata01 = a * bb + c * d
    ata11 = bb * bb + d * d

    atb0 = a * b[..., 0, 0] + c * b[..., 1, 0]
    atb1 = bb * b[..., 0, 0] + d * b[..., 1, 0]

    # λ from trace scale; ensure strictly positive even if A is zero
    tr = ata00 + ata11
    lam = (tr * 1e-8).clamp_min(eps)

    m00 = ata00 + lam
    m01 = ata01
    m10 = ata01
    m11 = ata11 + lam

    det_m = m00 * m11 - m01 * m10
    det_m_safe = det_m.abs().clamp_min(eps)
    inv_det_m = 1.0 / det_m_safe

    invm00 = m11 * inv_det_m
    invm01 = (-m01) * inv_det_m
    invm10 = (-m10) * inv_det_m
    invm11 = m00 * inv_det_m

    x0_fb = invm00 * atb0 + invm01 * atb1
    x1_fb = invm10 * atb0 + invm11 * atb1
    x_fb = torch.stack((x0_fb, x1_fb), dim=-1).unsqueeze(-1)  # (...,2,1)

    # choose fallback only when bad; else direct
    x = torch.where(bad.unsqueeze(-1).unsqueeze(-1), x_fb, x_dir)

    # if still non-finite, mark bad and zero it
    nonfinite = torch.isnan(x).flatten(-2).any(-1) | torch.isinf(x).flatten(-2).any(-1)
    bad = bad | nonfinite
    x = torch.where(bad.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(x), x)

    return x, bad


class _NullSpaceBasis(torch.autograd.Function):
    r"""The four right singular vectors of ``X`` with the smallest singular values, differentiably.

    ``torch.linalg.svd`` leaves the right singular vectors past ``min(N, 9)`` without a gradient, and
    for fewer than 9 points some or all of the four the 5-point solver uses are exactly those, so the
    backward pass silently dropped their gradient (all of it for ``N = 5``). This is used for
    ``N < 9`` only.

    The candidates depend only on the subspace the four vectors span, not on the basis chosen within
    it, so the backward differentiates the subspace. With
    :math:`X^\top X = \sum_j \lambda_j v_j v_j^\top`, each basis vector :math:`v_i` moves only out of
    the subspace :math:`S`, by
    :math:`dv_i = \sum_{j \notin S} v_j \, v_j^\top d(X^\top X) \, v_i / (\lambda_i - \lambda_j)`.
    That needs a gap between the fourth and fifth smallest singular values. Without one the subspace
    itself is not unique, and a nonzero incoming gradient gives a gradient that is not finite, as for
    ``torch.linalg.svd``; a zero incoming gradient, as from a sample whose candidates were discarded,
    gives zero. The forward pass is the same ``_torch_svd_cast`` call as before, so its result is
    unchanged.
    """

    # forward and setup_context are separate, so torch.func transforms (grad, vjp, jacrev) accept it
    @staticmethod
    def forward(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, S, V = _torch_svd_cast(X)  # V: (B, 9, 9)
        return V[:, :, -4:].contiguous(), S, V  # (B, 9, 4); S and V are returned only to be saved

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[torch.Tensor], output: Tuple[torch.Tensor, ...]) -> None:
        (X,) = inputs
        _, S, V = output
        ctx.mark_non_differentiable(S, V)
        ctx.save_for_backward(X, S, V)

    @staticmethod
    def backward(ctx: Any, grad_basis: torch.Tensor, _grad_S: Any, _grad_V: Any) -> torch.Tensor:
        X, S, V = ctx.saved_tensors
        work = torch.float64 if X.dtype == torch.float32 and X.device.type != "mps" else X.dtype
        X_, S_, V_, g = X.to(work), S.to(work), V.to(work), grad_basis.to(work)
        # eigenvalues of X^T X in the order of V's columns; the columns past min(N, 9) have eigenvalue 0
        lam = torch.cat((S_ * S_, S_.new_zeros(S_.shape[0], V_.shape[-1] - S_.shape[-1])), dim=-1)
        V_out, V_in = V_[:, :, :-4], V_[:, :, -4:]
        gap = lam[:, -4:].unsqueeze(-2) - lam[:, :-4].unsqueeze(-1)  # (B, 5, 4): lambda_i - lambda_j
        num = V_out.transpose(-1, -2) @ g
        # A zero incoming gradient contributes nothing, also where there is no gap: a sample whose
        # candidates were discarded then gets a zero gradient instead of 0 / 0.
        K = torch.where(num == 0, torch.zeros_like(num), num / gap)
        M = V_out @ K @ V_in.transpose(-1, -2)  # dL = <dG, M>
        return (X_ @ (M + M.transpose(-1, -2))).to(X.dtype)


def _least_squares_basis(design: torch.Tensor) -> torch.Tensor:
    """Least-squares null space ``(B, 9, 4)`` of design rows ``(B, N, 9)`` with ``N > 5``.

    The four right singular vectors with the smallest singular values, in
    :func:`~kornia.geometry.epipolar.fundamental._solve_dtype`.
    """
    design = design.to(_solve_dtype(design.device))
    if design.shape[-2] < 9:
        return _NullSpaceBasis.apply(design)[0]
    # every right singular vector has a gradient in torch.linalg.svd itself
    _, _, V = _torch_svd_cast(design)
    return V[:, :, -4:]


def _least_squares_candidates(design: torch.Tensor) -> torch.Tensor:
    """Nister's candidates ``(B, 10, 3, 3)`` of design rows ``(B, N, 9)`` with ``N > 5``, on the design's device.

    MPS inputs are solved on the host, as in :func:`_five_point_candidates`.
    """
    if design.device.type == "mps":
        return _least_squares_candidates(design.cpu()).to(design.device)
    return _nister_candidates(_least_squares_basis(design), design.dtype)[0]


def _five_point_candidates(design: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nister's five-point solver on the epipolar design rows ``(B, 5, 9)`` of minimal samples.

    The four-dimensional null space comes from batched Householder reflections,
    :func:`~kornia.geometry.solvers.homogeneous._null_space_householder`, and everything up to the roots is computed
    in float64, so a float32 sample does not lose its true solution to rounding.

    A sample whose design rows are rank deficient, such as one with a repeated correspondence or points collinear in
    both images, has a null space of more than four dimensions and no unique solution: all ten of its slots are NaN.
    It is found by a partial-pivoted LU factorization of the detached rows. Rounding rarely leaves an exactly zero
    pivot, so rank deficiency is a pivot below ``1e3`` epsilons of the largest one: repeated correspondences measure
    at most about 1e-16 relative, regular samples above 1e-8 even with all five points in a patch 1e-4 wide.
    Collinear points rounded to float32 are no longer collinear, and are solved like any other sample. A rank-deficient
    sample is swapped for a constant full-rank design before the null space is taken, whose normalizations would
    otherwise divide by a vanishing norm, so that its gradient is zero.

    Note:
        MPS has no float64, so an MPS sample is solved on the host and its candidates are copied back; autograd
        follows both copies. Solved in float32 on the device instead, an exact sample could miss its true solution
        by more than 1e-3, and on an Apple M1 (torch 2.14) the host solve was also faster: 8 ms against 37 ms for 256
        samples, 45 ms against 75 ms for 2048, and about the same at 8192.

    Returns:
        Candidates ``(B, 10, 3, 3)`` of unit Frobenius norm in the design's dtype, NaN where the slot holds no
        real root, and the ``(B, 10)`` mask of the real ones.
    """
    if design.device.type == "mps":
        candidates, valid = _five_point_candidates(design.cpu())
        return candidates.to(design.device), valid.to(design.device)
    A = design.to(_solve_dtype(design.device))
    lu, _, info = torch.linalg.lu_factor_ex(A.detach().mT)
    pivots = lu.diagonal(dim1=-2, dim2=-1).abs()
    rank_deficient = (info > 0) | (pivots.amin(-1) <= 1e3 * torch.finfo(A.dtype).eps * pivots.amax(-1))
    A = torch.where(rank_deficient[:, None, None], torch.eye(5, 9, device=A.device, dtype=A.dtype), A)
    candidates, valid = _nister_candidates(_null_space_householder(A), design.dtype)
    valid = valid & ~rank_deficient[:, None]
    return torch.where(valid[..., None, None], candidates, torch.full_like(candidates, float("nan"))), valid


# Monomial product tables of Nister's constraints: linear x linear -> quadratic (x^2, xy, xz, x, y^2, yz, y, z^2, z,
# 1), and quadratic x linear -> the twenty cubic monomials, in the order the elimination below expects.
_LINEAR_PRODUCTS = T_deg1.view(4, 4, 10)
_CUBIC_PRODUCTS = T_deg2.view(10, 4, 20)


def _scaled_polynomial_powers(roots: torch.Tensor) -> torch.Tensor:
    """Return ``z**d / max(1, abs(z))**10`` for ``d = 0, ..., 10`` without large powers.

    For ``abs(z) > 1`` these are the powers of ``1/z`` in reverse order (the degree ten is even).
    Every multiplication therefore has magnitude at most one. Cumulative products avoid the two general
    tensor exponentiations in the Newton correction, which dominate its CPU cost.
    """
    large = roots.abs() > 1.0
    bounded = torch.where(large, torch.where(large, roots, 1.0).reciprocal(), roots)
    powers = bounded[..., None].expand(*bounded.shape, 10).cumprod(-1)
    powers = torch.cat((torch.ones_like(bounded[..., None]), powers), -1)
    return torch.where(large[..., None], powers.flip(-1), powers)


def _nister_candidates(basis: torch.Tensor, out_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Essential matrices ``E = x X + y Y + z Z + W`` in the span of ``basis`` ``(B, 9, 4)`` (columns X, Y, Z, W).

    Nister's method: the cubic constraints ``det E = 0`` and ``2 E E^T E - tr(E E^T) E = 0`` as a ``(10, 20)``
    coefficient matrix, Gauss-Jordan elimination of its first ten monomials, a degree-ten polynomial in ``z`` from
    the determinant of the ``(3, 13)`` hidden-variable matrix, its real roots, and ``x``, ``y`` by
    back-substitution.

    The roots are the eigenvalues of the polynomial's companion matrix, from LAPACK on the host in float64 for every
    device: ``torch.linalg.eigvals`` has no batched CUDA kernel and none at all on MPS, and it is faster in
    float64 than in float32. They carry no gradient themselves: one Newton step on the differentiable polynomial
    polishes each root and supplies its derivative by the implicit function theorem, and is skipped at a multiple
    root, where the derivative does not exist.
    """
    B = basis.shape[0]
    device, dtype = basis.device, basis.dtype
    linear = _LINEAR_PRODUCTS.to(device, dtype)
    cubic = _CUBIC_PRODUCTS.to(device, dtype)
    N = basis.reshape(B, 3, 3, 4)  # entries of E as linear forms in (x, y, z, 1)

    # ---- constraints ----
    # det E, expanded along the third row: its cofactors are the cross product of the first two.
    rows01 = torch.einsum("bja,bkc,acm->bjkm", N[:, 0], N[:, 1], linear)
    cofactors = torch.stack(
        [rows01[:, 1, 2] - rows01[:, 2, 1], rows01[:, 2, 0] - rows01[:, 0, 2], rows01[:, 0, 1] - rows01[:, 1, 0]], 1
    )
    determinant = torch.einsum("bjm,bjc,mcn->bn", cofactors, N[:, 2], cubic)
    # 2 E E^T E - tr(E E^T) E = 2 (E E^T - tr(E E^T) I / 2) E
    EEt = torch.einsum("bika,bjkc,acm->bijm", N, N, linear)
    half_trace = 0.5 * EEt.diagonal(dim1=1, dim2=2).sum(-1)
    EEt = EEt - torch.eye(3, device=device, dtype=dtype)[None, :, :, None] * half_trace[:, None, None, :]
    trace_constraints = torch.einsum("bikm,bkjc,mcn->bijn", EEt, N, cubic).reshape(B, 9, 20)
    coeffs = torch.cat([trace_constraints, determinant[:, None]], 1)  # (B, 10, 20)

    # ---- elimination ----
    # An exactly singular elimination matrix means the sample has no solution. Its elements are factorized against
    # the identity instead, so that nothing below overflows or raises in the forward pass or gives a NaN gradient
    # in the backward, and all ten of their slots are NaN.
    A10, b10 = coeffs[..., :10], coeffs[..., 10:]
    eye10 = torch.eye(10, device=device, dtype=dtype)
    lu, pivots, info = torch.linalg.lu_factor_ex(A10.detach() if A10.requires_grad else A10)
    singular = info > 0
    if A10.requires_grad:
        # The backward of a singular factorization is 0 * inf even in the rows whose result is replaced (x86 LAPACK):
        # differentiate the factorization of the substituted matrix instead.
        lu, pivots, _ = torch.linalg.lu_factor_ex(torch.where(singular[:, None, None], eye10, A10))
    else:
        # The factorization of the identity is the identity without row exchanges: substituting it costs no second
        # factorization, and no host synchronization, which would split a compiled graph.
        lu = torch.where(singular[:, None, None], eye10, lu)
        pivots = torch.where(singular[:, None], torch.arange(1, 11, device=device, dtype=pivots.dtype), pivots)
    eliminated = torch.linalg.lu_solve(lu, pivots, b10)  # (B, 10, 10)

    # ---- hidden-variable matrix (B, 3, 13) and its determinant, a polynomial of degree ten in z ----
    top, bottom = eliminated[:, [4, 6, 8]], eliminated[:, [5, 7, 9]]
    A = torch.zeros(B, 3, 13, device=device, dtype=dtype)
    A[:, :, 1:4] = top[..., 0:3]
    A[:, :, 0:3] = A[:, :, 0:3] - bottom[..., 0:3]
    A[:, :, 5:8] = top[..., 3:6]
    A[:, :, 4:7] = A[:, :, 4:7] - bottom[..., 3:6]
    A[:, :, 9:13] = top[..., 6:10]
    A[:, :, 8:12] = A[:, :, 8:12] - bottom[..., 6:10]
    cs = _determinant_to_polynomial_jit(A)  # (B, 11), z^0 first

    # ---- real roots ----
    # A polynomial that is not finite, or whose leading coefficient is exactly zero (its degree is below ten, #4831),
    # has no usable companion matrix: its sample gets the identity instead and NaN slots.
    lead = cs[:, -1]
    usable = ~singular & torch.isfinite(cs).all(1) & (lead != 0)
    detached = cs.detach().cpu().double()
    companion = torch.zeros(B, 10, 10, dtype=torch.float64)
    companion[:, :9, 1:] = torch.eye(9, dtype=torch.float64)
    companion[:, 9] = -detached[:, :10] / torch.where(detached[:, 10:] == 0, 1.0, detached[:, 10:])
    host_usable = usable.cpu() & torch.isfinite(companion).flatten(1).all(1)
    companion = torch.where(host_usable[:, None, None], companion, torch.eye(10, dtype=torch.float64))
    eigenvalues = torch.linalg.eigvals(companion)
    real = eigenvalues.imag.abs() <= 1e-10 * eigenvalues.abs().clamp(min=1.0)
    # A root beyond the fifth root of the dtype's largest number could overflow the z^4 terms of the back-substitution,
    # and a masked infinity still turns the backward into 0 * inf = NaN: its slot is dropped, and a zero stands in for
    # it. That only binds for a float32 solve, at |z| > 5.5e7, where the basis matrix W no longer shows in E.
    in_range = eigenvalues.real.abs() < torch.finfo(dtype).max ** 0.2
    valid = (real & in_range & host_usable[:, None]).to(device)
    root0 = torch.where(real & in_range, eigenvalues.real, 0.0).to(device, dtype)

    # ---- Newton step: the root's polish and its gradient ----
    # The polynomial and its slope are evaluated divided by s^10 with s = max(1, |z|): the same step, without the
    # overflow of z^10 for large roots in float32.
    degrees = torch.arange(11, device=device, dtype=dtype)
    powers = _scaled_polynomial_powers(root0)  # (B, 10, 11); root0 carries no gradient
    value = (powers * cs[:, None]).sum(-1)
    slope_terms = powers[..., :10] * (degrees[1:] * cs[:, 1:])[:, None]
    slope = slope_terms.sum(-1)
    # At a multiple root rounding leaves a tiny slope: dividing two rounding errors would move an accurate root.
    simple = valid & (slope.abs() > 8 * torch.finfo(dtype).eps * slope_terms.abs().sum(-1))
    simple = simple & torch.isfinite(value) & torch.isfinite(slope)
    step = value / torch.where(simple, slope, torch.ones_like(slope))
    z = root0 - torch.where(simple, step, torch.zeros_like(step))

    # ---- back-substitution for x and y ----
    zz = z[:, None]  # (B, 1, 10)
    Bs = torch.stack(
        (
            A[:, :3, :1] * zz**3 + A[:, :3, 1:2] * zz.square() + A[:, :3, 2:3] * zz + A[:, :3, 3:4],
            A[:, :3, 4:5] * zz**3 + A[:, :3, 5:6] * zz.square() + A[:, :3, 6:7] * zz + A[:, :3, 7:8],
        ),
        dim=1,
    ).transpose(1, -1)  # (B, 10, 3, 2)
    bs = (
        (A[:, :3, 8:9] * zz**4 + A[:, :3, 9:10] * zz**3 + A[:, :3, 10:11] * zz.square() + A[:, :3, 11:12] * zz)
        + A[:, :3, 12:13]
    ).transpose(1, 2)[..., None]  # (B, 10, 3, 1)
    # Each 2x2 system is divided by its largest entry, which leaves its solution unchanged: entries of size A z^3 would
    # otherwise overflow the squares of the regularized fallback in float32, and a masked infinity still gives a NaN
    # gradient. The scale is detached; the solution does not depend on it.
    system = torch.cat([Bs[:, :, :2, :2].flatten(2), bs[:, :, :2, 0]], -1)
    system_scale = system.detach().abs().amax(-1).clamp(min=torch.finfo(dtype).tiny)[..., None, None]
    xy, bad2 = _solve_2x2_tikhonov_safe(Bs[:, :, :2, :2] / system_scale, bs[:, :, :2] / system_scale, 1e-12)
    coefficients = torch.stack([-xy[..., 0, 0], -xy[..., 1, 0], z, torch.ones_like(z)], -1)  # (B, 10, 4)
    Es = torch.einsum("bijk,brk->brij", N, coefficients)
    norm = Es.flatten(2).norm(dim=-1)[..., None, None]
    Es = Es / torch.where(norm > 0, norm, torch.ones_like(norm))
    valid = valid & ~bad2 & torch.isfinite(Es).flatten(2).all(-1)
    Es = torch.where(valid[..., None, None], Es, torch.full_like(Es, float("nan")))
    return Es.to(out_dtype), valid


def fun_select(null_mat: torch.Tensor, i: int, j: int, ratio: int = 3) -> torch.Tensor:
    return null_mat[:, ratio * j + i]


# vec(x1 x2^T) -> vec(x2 x1^T): the column order of the historical design rows in terms of _epipolar_design_rows'.
_TRANSPOSED_ROWS = [0, 3, 6, 1, 4, 7, 2, 5, 8]


def null_to_Nister_solution(X: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Candidates ``(B, 10, 3, 3)`` of design rows ``X`` ``(B, N, 9)`` in the column order ``vec(x1 x2^T)``."""
    design = X[..., _TRANSPOSED_ROWS]
    if design.shape[1] == 5:
        return _five_point_candidates(design)[0]
    return _least_squares_candidates(design)


def essential_from_fundamental(F_mat: torch.Tensor, K1: torch.Tensor, K2: torch.Tensor) -> torch.Tensor:
    r"""Get Essential matrix from Fundamental and Camera matrices.

    Uses the method from Hartley/Zisserman 9.6 pag 257 (formula 9.12).

    Convention:
        - :math:`E = K_2^\top F K_1` with ``K1`` the camera of the first image, so ``E`` follows the
          :math:`x_2^\top E x_1 = 0` order of :func:`find_essential`; it keeps the scale of ``F_mat``.

    Args:
        F_mat: The fundamental matrix with shape of :math:`(*, 3, 3)`.
        K1: The camera matrix from first camera with shape :math:`(*, 3, 3)`.
        K2: The camera matrix from second camera with shape :math:`(*, 3, 3)`.

    Returns:
        The essential matrix with shape :math:`(*, 3, 3)`.

    """
    KORNIA_CHECK_SHAPE(F_mat, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(K1, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(K2, ["*", "3", "3"])
    return K2.transpose(-2, -1) @ F_mat @ K1


def project_to_essential(E_mat: torch.Tensor) -> torch.Tensor:
    r"""Project a matrix onto the essential-matrix manifold.

    An essential matrix must be of the form :math:`U \text{diag}(s, s, 0) V^T`. Matrices estimated
    by other means (e.g. a fundamental matrix fitted with the 8-point DLT) generally violate this
    constraint, which silently breaks tools that rely on it, such as
    :func:`decompose_essential_matrix` and :func:`motion_from_essential`. The projection averages
    the two largest singular values and zeroes the smallest one.

    Convention:
        - The result is not normalised to unit Frobenius norm, unlike the candidates of :func:`find_essential`.

    Args:
        E_mat: The matrices to project with shape :math:`(*, 3, 3)`.

    Returns:
        The closest matrices (in Frobenius norm) satisfying the essential-matrix constraint,
        with shape :math:`(*, 3, 3)`.

    """
    KORNIA_CHECK_SHAPE(E_mat, ["*", "3", "3"])
    U, S, V = _torch_svd_cast(E_mat)
    S_new = torch.zeros_like(S)
    mean_sv = 0.5 * (S[..., 0] + S[..., 1])
    S_new[..., 0] = mean_sv
    S_new[..., 1] = mean_sv
    return U @ torch.diag_embed(S_new) @ V.mH


def decompose_essential_matrix(E_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Decompose an essential matrix to possible rotations and translation.

    This function decomposes the essential matrix E using svd decomposition [96]
    and give the possible solutions: :math:`R1, R2, t`.

    Convention:
        - Returns two rotations and a unit translation; the true pose is one of :math:`(R_1, \pm t)`,
          :math:`(R_2, \pm t)`. :math:`(R_1, -t)` and :math:`(R_2, t)` give :math:`[t]_\times R` with the sign
          of ``E_mat``, :math:`(R_1, t)` and :math:`(R_2, -t)` the opposite sign. Which of the four is the true
          pose is not fixed: it can change when ``E_mat`` is negated or rescaled. Select by cheirality with
          :func:`motion_from_essential_choose_solution`; :ref:`two-view geometry <two-view-conventions>`
          compares OpenCV's labels.

    Args:
       E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
       A tuple containing the first and second possible rotation matrices and the translation vector,
       with shapes :math:`[(*, 3, 3), (*, 3, 3), (*, 3, 1)]`: the same leading dims as the input.

    """
    KORNIA_CHECK_SHAPE(E_mat, ["*", "3", "3"])

    # decompose matrix by its singular values
    U, _, V = _torch_svd_cast(E_mat)
    Vt = V.transpose(-2, -1)

    mask = torch.ones_like(E_mat)
    mask[..., -1:] *= -1.0  # fill last column with negative values

    maskt = mask.transpose(-2, -1)

    # avoid singularities
    U = torch.where((torch.det(U) < 0.0)[..., None, None], U * mask, U)
    Vt = torch.where((torch.det(Vt) < 0.0)[..., None, None], Vt * maskt, Vt)

    # W = [e_z]_x + diag(0, 0, 1), built unbatched so that it does not add a batch dim to an unbatched U.
    W = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=E_mat.device, dtype=E_mat.dtype)

    # reconstruct rotations and retrieve translation vector
    U_W_Vt = U @ W @ Vt
    U_Wt_Vt = U @ W.transpose(-2, -1) @ Vt

    # return values
    R1 = U_W_Vt
    R2 = U_Wt_Vt
    T = U[..., -1:]
    return (R1, R2, T)


def decompose_essential_matrix_no_svd(E_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Decompose the essential matrix to rotation and translation.

    Recovers the rotations and translation from an essential matrix without SVD.
    Reference: Horn, Berthold KP. Recovering baseline and orientation from essential matrix[J].
    J. Opt. Soc. Am, 1990, 110.

    Convention:
        - Same candidate set as :func:`decompose_essential_matrix`, with ``t`` of unit norm.

    Args:
       E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
       A tuple containing the first and second possible rotation matrices and the translation vector, with
       shapes :math:`[(B, 3, 3), (B, 3, 3), (B, 3, 1)]`: all leading dims are flattened into one batch dim.

    """
    KORNIA_CHECK_SHAPE(E_mat, ["*", "3", "3"])
    if len(E_mat.shape) != 3:
        E_mat = E_mat.view(-1, 3, 3)

    B = E_mat.shape[0]

    # Eq.18, choose the largest of the three possible pairwise cross-products
    e1, e2, e3 = E_mat[..., 0], E_mat[..., 1], E_mat[..., 2]

    # sqrt(1/2 trace(EE^T)), B
    scale_factor = torch.sqrt(0.5 * torch.diagonal(E_mat @ E_mat.transpose(-1, -2), dim1=-1, dim2=-2).sum(-1))

    # B, 3, 3
    cross_products = torch.stack(
        [torch.linalg.cross(e1, e2, dim=-1), torch.linalg.cross(e2, e3, dim=-1), torch.linalg.cross(e3, e1, dim=-1)],
        dim=1,
    )

    # B, 3, 1
    norms = torch.norm(cross_products, dim=-1, keepdim=True)

    # B, to select which b1
    largest = torch.argmax(norms, dim=-2)

    # B, 3, 3
    e_cross_products = scale_factor[:, None, None] * cross_products / norms

    # broadcast the index
    index_expanded = largest.unsqueeze(-1).expand(-1, -1, e_cross_products.size(-1))

    # slice at dim=1, select for each batch one b (e1*e2 or e2*e3 or e3*e1), B, 1, 3
    b1 = torch.gather(e_cross_products, dim=1, index=index_expanded).squeeze(1)
    # normalization
    b1_ = b1 / torch.norm(b1, dim=-1, keepdim=True)

    # skew-symmetric matrix
    B1 = torch.zeros((B, 3, 3), device=E_mat.device, dtype=E_mat.dtype)
    t0, t1, t2 = b1[:, 0], b1[:, 1], b1[:, 2]
    B1[:, 0, 1], B1[:, 1, 0] = -t2, t2
    B1[:, 0, 2], B1[:, 2, 0] = t1, -t1
    B1[:, 1, 2], B1[:, 2, 1] = -t0, t0

    # the second translation and rotation
    B2 = -B1
    b2 = -b1

    # Eq.24, recover R
    # (bb)R = Cofactors(E)^T - BE
    R1 = (matrix_cofactor_tensor(E_mat) - B1 @ E_mat) / (b1 * b1).sum(-1)[:, None, None]
    R2 = (matrix_cofactor_tensor(E_mat) - B2 @ E_mat) / (b2 * b2).sum(-1)[:, None, None]

    return (R1, R2, b1_.unsqueeze(-1))


def essential_from_Rt(R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    r"""Get the Essential matrix from Camera motion (Rs and ts).

    Reference: Hartley/Zisserman 9.6 pag 257 (formula 9.12)

    Convention:
        - Returns :math:`[t]_\times R` for the relative motion ``(R, t)`` of :func:`relative_camera_motion`, so
          :math:`x_2^\top E x_1 = 0` in normalised camera coordinates. It is not normalised:
          :math:`\|E\|_F = \sqrt{2} \|t\|`.

    Args:
        R1: The first camera rotation matrix with shape :math:`(*, 3, 3)`.
        t1: The first camera translation vector with shape :math:`(*, 3, 1)`.
        R2: The second camera rotation matrix with shape :math:`(*, 3, 3)`.
        t2: The second camera translation vector with shape :math:`(*, 3, 1)`.

    Returns:
        The Essential matrix with the shape :math:`(*, 3, 3)`.

    """
    KORNIA_CHECK_SHAPE(R1, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(R2, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(t1, ["*", "3", "1"])
    KORNIA_CHECK_SHAPE(t2, ["*", "3", "1"])

    # first compute the camera relative motion
    R, t = relative_camera_motion(R1, t1, R2, t2)

    # get the cross product from relative translation vector
    Tx = cross_product_matrix(t[..., 0])

    return Tx @ R


def motion_from_essential(E_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Get Motion (R's and t's ) from Essential matrix.

    Computes and return four possible poses exist for the decomposition of the Essential
    matrix. The possible solutions are :math:`[R1,t], [R1,-t], [R2,t], [R2,-t]`.

    Convention:
        - The candidates of :func:`decompose_essential_matrix` are stacked on dim ``-3`` in that order; which
          index is the true pose is not fixed.

    Args:
        E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
        The rotation and translation containing the four possible combination for the retrieved motion.
        The tuple is as following :math:`[(*, 4, 3, 3), (*, 4, 3, 1)]`.

    """
    KORNIA_CHECK_SHAPE(E_mat, ["*", "3", "3"])

    # decompose the essential matrix by its possible poses
    R1, R2, t = decompose_essential_matrix(E_mat)

    # compbine and returns the four possible solutions
    Rs = torch.stack([R1, R1, R2, R2], dim=-3)
    Ts = torch.stack([t, -t, t, -t], dim=-3)

    return Rs, Ts


def motion_from_essential_choose_solution(
    E_mat: torch.Tensor,
    K1: torch.Tensor,
    K2: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Recover the relative camera rotation and the translation from an estimated essential matrix.

    The method checks the corresponding points in two images and also returns the triangulated
    3d points. Internally uses :py:meth:`~kornia.geometry.epipolar.decompose_essential_matrix` and
    :py:meth:`~kornia.geometry.epipolar.triangulate_points`.

    Convention:
        - ``K1`` and ``K2`` are applied inside, so ``x1`` and ``x2`` are pixel coordinates.
        - Returns the candidate with the most points at positive depth in both cameras, with
          :math:`\|t\| = 1`, and the points triangulated in the first camera's frame at that scale.
        - Known defects: with no valid point it returns candidate 0 without a signal
          (`#4879 <https://github.com/kornia/kornia/issues/4879>`_).

    Args:
        E_mat: The essential matrix in the form of :math:`(B, 3, 3)`, or :math:`(3, 3)` with every other input
            unbatched too.
        K1: The camera matrix from first camera with shape :math:`(B, 3, 3)`.
        K2: The camera matrix from second camera with shape :math:`(B, 3, 3)`.
        x1: The set of points in the first image with shape :math:`(B, N, 2)`.
        x2: The set of points in the second image with shape :math:`(B, N, 2)`.
        mask: A boolean mask which can be used to exclude some points from choosing
          the best solution. This is useful for using this function with sets of points of
          different cardinality (for instance after filtering with RANSAC) while keeping batch
          semantics. Mask is of shape :math:`(B, N)`.

    Returns:
        The rotation and translation plus the 3d triangulated points.
        The tuple is as following :math:`[(B, 3, 3), (B, 3, 1), (B, N, 3)]`, without ``B`` for unbatched inputs.

    """
    KORNIA_CHECK_SHAPE(E_mat, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(K1, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(K2, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(x1, ["*", "N", "2"])
    KORNIA_CHECK_SHAPE(x2, ["*", "N", "2"])
    KORNIA_CHECK(len(E_mat.shape[:-2]) == len(K1.shape[:-2]) == len(K2.shape[:-2]))

    if mask is not None:
        KORNIA_CHECK_SHAPE(mask, ["*", "N"])
        KORNIA_CHECK(mask.shape == x1.shape[:-1])

    unbatched = len(E_mat.shape) == 2

    if unbatched:
        # add a leading batch dimension. We will remove it at the end, before
        # returning the results
        E_mat = E_mat[None]
        K1 = K1[None]
        K2 = K2[None]
        x1 = x1[None]
        x2 = x2[None]
        if mask is not None:
            mask = mask[None]

    # compute four possible pose solutions
    Rs, ts = motion_from_essential(E_mat)

    # set reference view pose and compute projection matrix
    R1 = eye_like(3, E_mat)  # Bx3x3
    t1 = vec_like(3, E_mat)  # Bx3x1

    # compute the projection matrices for first camera
    R1 = R1[:, None].expand(-1, 4, -1, -1)
    t1 = t1[:, None].expand(-1, 4, -1, -1)
    K1 = K1[:, None].expand(-1, 4, -1, -1)
    P1 = projection_from_KRt(K1, R1, t1)  # 1x4x4x4

    # compute the projection matrices for second camera
    R2 = Rs
    t2 = ts
    K2 = K2[:, None].expand(-1, 4, -1, -1)
    P2 = projection_from_KRt(K2, R2, t2)  # Bx4x4x4

    # triangulate the points
    x1 = x1[:, None].expand(-1, 4, -1, -1)
    x2 = x2[:, None].expand(-1, 4, -1, -1)
    X = triangulate_points(P1, P2, x1, x2)  # Bx4xNx3

    # project points and compute their depth values
    d1 = depth_from_point(R1, t1, X)
    d2 = depth_from_point(R2, t2, X)

    # verify the point values that have a positive depth value
    depth_mask = (d1 > 0.0) & (d2 > 0.0)
    if mask is not None:
        depth_mask &= mask.unsqueeze(1)

    mask_indices = torch.max(depth_mask.sum(-1), dim=-1, keepdim=True)[1]

    # get pose and points 3d and return
    batch_idx = torch.arange(mask_indices.shape[0], device=mask_indices.device)
    R_out = Rs[batch_idx, mask_indices[:, 0]]
    t_out = ts[batch_idx, mask_indices[:, 0]]
    points3d_out = X[batch_idx, mask_indices[:, 0]]

    if unbatched:
        R_out = R_out[0]
        t_out = t_out[0]
        points3d_out = points3d_out[0]

    return R_out, t_out, points3d_out


def relative_camera_motion(
    R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the relative camera motion between two cameras.

    Given the motion parameters of two cameras, computes the motion parameters of the second
    one assuming the first one to be at the origin. If :math:`T1` and :math:`T2` are the camera motions,
    the computed relative motion is :math:`T = T_{2}T^{-1}_{1}`.

    Convention:
        - Inputs are world-to-camera extrinsics, :math:`x_{cam} = R X + t` (see :doc:`camera and world
          conventions </get-started/camera-conventions>`); the result is :math:`R = R_2 R_1^\top` and
          :math:`t = t_2 - R_2 R_1^\top t_1`, the motion from camera 1 to camera 2. As 4x4 matrices this is
          :math:`E_2 E_1^{-1}`, whereas :func:`~kornia.geometry.linalg.relative_transformation` of :math:`E_1` and
          :math:`E_2` returns :math:`E_1^{-1} E_2`.

    Args:
        R1: The first camera rotation matrix with shape :math:`(*, 3, 3)`.
        t1: The first camera translation vector with shape :math:`(*, 3, 1)`.
        R2: The second camera rotation matrix with shape :math:`(*, 3, 3)`.
        t2: The second camera translation vector with shape :math:`(*, 3, 1)`.

    Returns:
        A tuple with the relative rotation matrix and
        translation vector with the shape of :math:`[(*, 3, 3), (*, 3, 1)]`.

    """
    KORNIA_CHECK_SHAPE(R1, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(R2, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(t1, ["*", "3", "1"])
    KORNIA_CHECK_SHAPE(t2, ["*", "3", "1"])

    # compute first the relative rotation
    R = R2 @ R1.transpose(-2, -1)

    # compute the relative translation vector
    t = t2 - R @ t1

    return R, t


def find_essential(
    points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Find essential matrices.

    Convention:
        - ``points1`` (first image) and ``points2`` (second image) are normalised camera coordinates
          :math:`K^{-1} [u, v, 1]^\top`, not pixels; each real candidate satisfies :math:`x_2^\top E x_1 = 0`
          in them. :ref:`Two-view geometry <two-view-conventions>` maps this onto OpenCV.
        - All ten slots are always returned: each real root gives a candidate of unit Frobenius norm, and each
          complex root a ``NaN`` slot, so a sample with no real solution returns ten ``NaN`` slots.
        - The solve runs in float64 whatever the input dtype, and the degree-ten polynomial's roots come from LAPACK
          on the host for every device. MPS has no float64, so MPS inputs are solved on the host and the candidates
          copied back, which on an Apple M1 was also faster than a float32 solve on the device. A sample whose five
          design rows are rank deficient, such as one with a repeated correspondence, has no unique solution: ten
          ``NaN`` slots and a zero gradient.
        - Known defects: ``weights`` is ignored (`#4876 <https://github.com/kornia/kornia/issues/4876>`_).

    Args:
         points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=5`.
         points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=5`.
         weights: Accepted with a shape of :math:`(B, N)` and ignored (see Known defects).

    Returns:
         the computed essential matrices with shape :math:`(B, 10, 3, 3)`.
         To choose the best one out of 10, try to check the one with the lowest Sampson distance, ignoring the
         ``NaN`` slots.

    """
    return run_5point(points1, points2, weights).to(points1.dtype)


def _refine_essential_lm(
    E: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    mask: Optional[torch.Tensor],
    loss: str,
    scale2: Union[float, torch.Tensor],
    iters: int,
) -> torch.Tensor:
    """Levenberg-Marquardt on the Sampson distance, batched over essential matrices ``(K, 3, 3)``.

    ``E = U diag(1, 1, 0) V^T / sqrt(2)`` with ``U`` and ``V`` in SO(3): the factorization of
    :func:`~kornia.geometry.epipolar.fundamental._refine_fundamental_lm` with its singular-value ratio fixed to one.
    Five parameters, as many as the rotation and translation direction of PoseLib's ``refine_relpose``: Cayley
    rotations of ``U`` about the three axes and of ``V`` about ``V e_1`` and ``V e_2``. Rotating both about their third
    axes by the same angle leaves ``E`` unchanged, so that direction is left out and the normal equations stay regular.
    Residuals, losses, masks and step acceptance as for the fundamental matrix; ``x1`` and ``x2`` are homogeneous
    ``(N, 3)`` calibrated points. Returns matrices of unit Frobenius norm. For RANSAC, under ``torch.no_grad``.
    """
    K = E.shape[0]
    dtype, device = E.dtype, E.device
    cpu = K > 0 and device.type == "cpu" and not torch.is_grad_enabled()
    if cpu and K == 1 and mask is not None and mask.dtype == torch.bool:
        x1, x2, mask = x1[mask[0]], x2[mask[0]], None
    H = _hat_basis(dtype, device)
    eye3 = torch.eye(3, dtype=dtype, device=device)
    eye5 = torch.eye(5, dtype=dtype, device=device)
    algebraic = _epipolar_design_rows(x1, x2).T  # (9, N)
    quadratic = torch.cat([_epipolar_design_rows(x1, x1), _epipolar_design_rows(x2, x2)], 1).T  # (18, N)
    U, _, Vh = torch.linalg.svd(E)
    V = Vh.mT
    # Proper rotations: the third singular vectors do not enter E, so their signs are free.
    U = torch.cat([U[..., :2], U[..., 2:] * torch.linalg.det(U).sign()[:, None, None]], -1)
    V = torch.cat([V[..., :2], V[..., 2:] * torch.linalg.det(V).sign()[:, None, None]], -1)
    damping = torch.full((K, 1, 1), 1e-3, dtype=dtype, device=device)

    def compose(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return (U[..., :2] @ V[..., :2].mT) * 0.5**0.5

    def normal_equations(E: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Directions dE/dp: H_a E (rotations of U), -E [V e_k]_x (rotations of V about its first two axes).
        axes = (V[..., :2].mT.reshape(K * 2, 3) @ H.reshape(3, 9)).reshape(K, 2, 3, 3)
        tangent = torch.cat([H @ E[:, None], (E[:, None] @ axes).neg()], 1)
        return _sampson_normal_equations(E, tangent, algebraic, quadratic, mask, loss, scale2)

    def cayley(half: torch.Tensor) -> torch.Tensor:
        # The rotation by the vector 2 * half: Cayley transform of the half-angle skew matrix.
        skew = (half @ H.reshape(3, 9)).reshape(-1, 3, 3)
        factor = (2.0 / (1.0 + half.square().sum(1)))[:, None, None]
        return eye3 + factor * (skew + skew @ skew)

    E = compose(U, V)
    system, cost = normal_equations(E, V)
    for iteration in range(iters):
        delta = -torch.linalg.solve_ex(system[..., :5] + damping * eye5, system[..., 5:])[0][..., 0]
        U_new = cayley(0.5 * delta[:, :3]) @ U
        V_new = cayley(0.5 * (V[..., :2] @ delta[:, 3:, None])[..., 0]) @ V
        E_new = compose(U_new, V_new)
        if cpu and iteration + 1 == iters:
            cost_new = _sampson_cost(E_new, algebraic, quadratic, mask, loss, scale2)
            accepted = cost_new < cost
            return torch.where(accepted[:, None, None], E_new, E)
        system_new, cost_new = normal_equations(E_new, V_new)
        accept = (cost_new < cost)[:, None, None]
        U, V = torch.where(accept, U_new, U), torch.where(accept, V_new, V)
        E, system = torch.where(accept, E_new, E), torch.where(accept, system_new, system)
        cost = torch.where(accept[:, 0, 0], cost_new, cost)
        damping = damping * torch.where(accept, 0.1, 10.0)
    return E
