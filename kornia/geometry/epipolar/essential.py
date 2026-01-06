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

from typing import Optional, Tuple

import torch

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SAME_SHAPE, KORNIA_CHECK_SHAPE
from kornia.core.ops import eye_like, vec_like
from kornia.core.utils import _torch_svd_cast
from kornia.geometry.solvers.polynomial_solver import T_deg1, T_deg2, coefficient_map, multiplication_indices, signs

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
    "relative_camera_motion",
]


def run_5point(points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Compute the essential matrix using the 5-point algorithm from Nister.

    The linear system is solved by Nister's 5-point algorithm [@nister2004efficient],
    and the solver implemented referred to [@barath2020magsac++][@wei2023generalized][@wang2023vggsfm].

    Args:
        points1: A set of carlibrated points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Not used, kept for compatibility.

    Returns:
        the computed essential matrix with shape :math:`(B, 10, 3, 3)`.

    """
    KORNIA_CHECK_SHAPE(points1, ["B", "N", "2"])
    KORNIA_CHECK_SAME_SHAPE(points1, points2)
    KORNIA_CHECK(points1.shape[1] >= 5, "Number of points should be >=5")

    batch_size, _, _ = points1.shape
    x1, y1 = points1[..., 0:1], points1[..., 1:2]
    x2, y2 = points2[..., 0:1], points2[..., 1:2]
    ones = torch.ones_like(x1)
    # build the equation system and find the null space.
    # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf
    # [x * x', x * y', x, y * x', y * y', y, x', y', 1]
    # BxNx9
    X = torch.cat([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, ones], dim=-1)
    # use Nister's 5PC to solve essential matrix
    E = null_to_Nister_solution(X, batch_size)
    bad = torch.isnan(E).all(dim=(-1, -2)).all(dim=-1)  # (B,)
    if bad.any():
        eye3 = torch.eye(3, device=E.device, dtype=E.dtype).view(1, 1, 3, 3).expand(batch_size, 10, 3, 3)
        E = torch.where(bad.view(batch_size, 1, 1, 1), eye3, E)
    return E


@torch.jit.script
def _multiply_deg_one_poly(a: torch.Tensor, b: torch.Tensor, T_deg1: torch.Tensor) -> torch.Tensor:
    # a, b: (..., 4)
    product_basis = a.unsqueeze(2) * b.unsqueeze(1)  # (..., 4, 4)
    product_vector = product_basis.flatten(start_dim=-2)  # (..., 16)
    return product_vector @ T_deg1  # (..., 10)


@torch.jit.script
def _multiply_deg_two_one_poly(a: torch.Tensor, b: torch.Tensor, T_deg2: torch.Tensor) -> torch.Tensor:
    # a: (..., 10), b: (..., 4)
    product_basis = a.unsqueeze(2) * b.unsqueeze(1)  # (..., 10, 4)
    product_vector = product_basis.flatten(start_dim=-2)  # (..., 40)
    return product_vector @ T_deg2  # (..., 20)


@torch.jit.script
def _determinant_to_polynomial_jit(
    A: torch.Tensor,
    multiplication_indices: torch.Tensor,
    signs: torch.Tensor,
    coefficient_map: torch.Tensor,
) -> torch.Tensor:
    # A: (B, 3, 13) -> (B, 11)
    B = A.shape[0]
    A_flat = A.view(B, -1)  # (B, 39)

    gathered_values = A_flat[:, multiplication_indices]  # (B, 486, 3)
    products = torch.prod(gathered_values, dim=-1)  # (B, 486)
    signed_products = products * signs  # (B, 486)

    cs = torch.zeros(B, 11, device=A.device, dtype=A.dtype)
    batch_coefficient_map = coefficient_map.unsqueeze(0).expand(B, -1)  # (B, 486)
    cs.scatter_add_(1, batch_coefficient_map, signed_products)
    return cs


@torch.jit.script
def _solve_2x2_tikhonov_safe(A: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Solve (A)x=b for A (...,2,2), b (...,2,1) using:
      - direct inverse when det is OK
      - otherwise solve normal equations (A^T A + λI)x = A^T b  (λ from trace scale)
    Never throws. Returns (x, bad) where bad marks ill-conditioned A.
    Args:
        A: (...,2,2)
        b: (...,2,1)
        eps: small value to avoid unstabilities.
    Returns:
        x: (...,2,1)
        bad: (...,2,1)
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
    nonfinite = torch.isnan(x).any(dim=(-2, -1)) | torch.isinf(x).any(dim=(-2, -1))
    bad = bad | nonfinite
    x = torch.where(bad.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(x), x)

    return x, bad


# --- fun_select inline, to keep it JIT-simple ---
def _fun_select(mat: torch.Tensor, i: int, j: int, ratio: int = 3) -> torch.Tensor:
    return mat[:, ratio * j + i]


@torch.jit.script
def _null_to_Nister_solution_script(
    X: torch.Tensor,
    batch_size: int,
    T_deg1: torch.Tensor,
    T_deg2: torch.Tensor,
    multiplication_indices: torch.Tensor,
    signs: torch.Tensor,
    coefficient_map: torch.Tensor,
    idx_ij: torch.Tensor,  # kept for signature compatibility (unused)
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    idx3: torch.Tensor,
    top_idx: torch.Tensor,
    bot_idx: torch.Tensor,
) -> torch.Tensor:
    original_dtype = X.dtype

    _, _, V = _torch_svd_cast(X)  # V: (B, 9, 9)
    null_ = V[:, :, -4:].contiguous()  # (B, 9, 4)
    nullSpace = V.transpose(-1, -2)[:, -4:, :]  # (B, 4, 9)

    B = batch_size
    device = X.device
    dtype = X.dtype

    coeffs = torch.zeros(B, 10, 20, device=device, dtype=dtype)

    # (B,9,4) -> (B,3,3,4) with column-major fix
    null_ij = null_.view(B, 3, 3, 4).transpose(1, 2).contiguous()  # (B, 3, 3, 4)

    # ---- determinant constraint ----
    n00 = null_ij[:, 0, 0, :]
    n01 = null_ij[:, 0, 1, :]
    n02 = null_ij[:, 0, 2, :]
    n10 = null_ij[:, 1, 0, :]
    n11 = null_ij[:, 1, 1, :]
    n12 = null_ij[:, 1, 2, :]
    n20 = null_ij[:, 2, 0, :]
    n21 = null_ij[:, 2, 1, :]
    n22 = null_ij[:, 2, 2, :]

    # small reuse to reduce launches
    p01_12 = _multiply_deg_one_poly(n01, n12, T_deg1)
    p02_11 = _multiply_deg_one_poly(n02, n11, T_deg1)
    p02_10 = _multiply_deg_one_poly(n02, n10, T_deg1)
    p00_12 = _multiply_deg_one_poly(n00, n12, T_deg1)
    p00_11 = _multiply_deg_one_poly(n00, n11, T_deg1)
    p01_10 = _multiply_deg_one_poly(n01, n10, T_deg1)

    coeffs[:, 9] = (
        _multiply_deg_two_one_poly(p01_12 - p02_11, n20, T_deg2)
        + _multiply_deg_two_one_poly(p02_10 - p00_12, n21, T_deg2)
        + _multiply_deg_two_one_poly(p00_11 - p01_10, n22, T_deg2)
    )

    # ---- EE^T constraints ----
    a_data = null_ij[:, i_idx, :, :]  # (B,9,3,4)
    b_data = null_ij[:, j_idx, :, :]  # (B,9,3,4)
    prods = _multiply_deg_one_poly(a_data.reshape(-1, 4), b_data.reshape(-1, 4), T_deg1).view(B, 9, 3, 10)
    D_blocks = prods.sum(dim=2).view(B, 3, 3, 10).contiguous()

    # trace removal
    diag = D_blocks[:, idx3, idx3, :]  # (B,3,10)
    t = 0.5 * diag.sum(dim=1, keepdim=True)  # (B,1,10)
    D_blocks[:, idx3, idx3, :] = D_blocks[:, idx3, idx3, :] - t

    # first 9 rows of coeffs
    D_for_i = D_blocks[:, i_idx, :, :]  # (B,9,3,10)
    Null_for_j = null_ij[:, :, j_idx, :].permute(0, 2, 1, 3).contiguous()  # (B,9,3,4)
    prods2 = _multiply_deg_two_one_poly(D_for_i.reshape(-1, 10), Null_for_j.reshape(-1, 4), T_deg2).view(B, 9, 3, 20)
    coeffs[:, :9, :] = prods2.sum(dim=2)  # (B,9,20)

    # ---- elimination: solve A10 * X = b_poly ----
    A10 = coeffs[:, :, :10]  # (B, 10, 10)
    b_poly = coeffs[:, :, 10:]  # (B, 10, 10)

    # Prefer direct solve; if singular, add tiny damping (no batch compaction).
    eye10 = torch.eye(10, device=device, dtype=dtype).unsqueeze(0).expand(B, 10, 10)

    # Try direct solve first
    eliminated = torch.linalg.solve(A10, b_poly)  # (B,10,10)

    # Detect NaN/Inf from singular solve and fix with damping solve
    bad = torch.isnan(eliminated).any(dim=(-2, -1)) | torch.isinf(eliminated).any(dim=(-2, -1))  # (B,)
    if bad.any():
        # Damped solve only for bad rows but WITHOUT compaction:
        # build damped A = A10 + λI, where λ depends on scale
        # (use per-batch scalar to avoid huge allocations)
        diagA = torch.diagonal(A10, dim1=-2, dim2=-1).abs().mean(dim=-1)  # (B,)
        lam = (diagA * 1e-8 + 1e-8).to(dtype)  # (B,)
        A_damped = A10 + eye10 * lam.view(B, 1, 1)
        eliminated_d = torch.linalg.solve(A_damped, b_poly)
        eliminated = torch.where(bad.view(B, 1, 1), eliminated_d, eliminated)

    coeffs_ = torch.cat((A10, eliminated), dim=-1)  # (B,10,20)

    # ---- build A (B,3,13) ----
    A = torch.zeros(B, 3, 13, device=device, dtype=dtype)

    A[:, :, 1:4] = coeffs_[:, top_idx, 10:13]
    A[:, :, 0:3] = A[:, :, 0:3] - coeffs_[:, bot_idx, 10:13]

    A[:, :, 5:8] = coeffs_[:, top_idx, 13:16]
    A[:, :, 4:7] = A[:, :, 4:7] - coeffs_[:, bot_idx, 13:16]

    A[:, :, 9:13] = coeffs_[:, top_idx, 16:20]
    A[:, :, 8:12] = A[:, :, 8:12] - coeffs_[:, bot_idx, 16:20]

    # ---- determinant polynomial -> companion ----
    cs = _determinant_to_polynomial_jit(A, multiplication_indices, signs, coefficient_map)  # (B,11)

    C = torch.zeros(B, 10, 10, device=device, dtype=dtype)
    C[:, 0:9, 1:10] = torch.eye(9, device=device, dtype=dtype)

    cs_de = cs[:, -1].clamp_min(1e-8)
    C[:, -1, :] = -cs[:, :-1] / cs_de.unsqueeze(-1)

    roots_eig = torch.linalg.eigvals(C)  # (B,10), complex
    roots = torch.real(roots_eig)
    is_real = torch.abs(torch.imag(roots_eig)) < 1e-10

    roots_unsqu = roots.unsqueeze(1)  # (B,1,10)

    Bs = torch.stack(
        (
            A[:, :3, :1] * (roots_unsqu**3)
            + A[:, :3, 1:2] * roots_unsqu.square()
            + A[:, :3, 2:3] * roots_unsqu
            + A[:, :3, 3:4],
            A[:, :3, 4:5] * (roots_unsqu**3)
            + A[:, :3, 5:6] * roots_unsqu.square()
            + A[:, :3, 6:7] * roots_unsqu
            + A[:, :3, 7:8],
        ),
        dim=1,
    ).transpose(1, -1)  # (B,10,3,2)

    bs_vec = (
        (
            A[:, :3, 8:9] * (roots_unsqu**4)
            + A[:, :3, 9:10] * (roots_unsqu**3)
            + A[:, :3, 10:11] * roots_unsqu.square()
            + A[:, :3, 11:12] * roots_unsqu
            + A[:, :3, 12:13]
        )
        .transpose(1, 2)
        .unsqueeze(-1)
    )  # (B,10,3,1)

    A2 = Bs[:, :, 0:2, 0:2]  # (B,10,2,2)
    b2 = bs_vec[:, :, 0:2, :]  # (B,10,2,1)

    xzs, bad2 = _solve_2x2_tikhonov_safe(A2, b2, 1e-12)  # never throws

    # ---- build Es ----
    xzs_sq = xzs.squeeze(-1)  # (B,10,2)
    x = -xzs_sq[:, :, 0]
    y = -xzs_sq[:, :, 1]
    z = roots

    N0 = nullSpace[:, 0, :].unsqueeze(1)  # (B,1,9)
    N1 = nullSpace[:, 1, :].unsqueeze(1)
    N2 = nullSpace[:, 2, :].unsqueeze(1)
    N3 = nullSpace[:, 3, :].unsqueeze(1)

    Es_vec = x.unsqueeze(-1) * N0 + y.unsqueeze(-1) * N1 + z.unsqueeze(-1) * N2 + N3  # (B,10,9)
    inv_norm = torch.rsqrt(x * x + y * y + z * z + 1.0)
    Es_vec = Es_vec * inv_norm.unsqueeze(-1)

    Es = Es_vec.view(B, 10, 3, 3).transpose(-1, -2)
    # after Es is created (B,10,3,3)
    if bad2.any():
        Es[bad2] = torch.nan
    # mark complex roots as NaN (keeps shape, no compaction)
    if is_real.logical_not().any():
        Es[~is_real] = torch.nan

    return Es.to(dtype=original_dtype)


# Indices for null_ij reshaping
IDX_IJ = torch.tensor([[0, 3, 6], [1, 4, 7], [2, 5, 8]], dtype=torch.long)
I_IDX = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
J_IDX = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)
IDX3 = torch.tensor([0, 1, 2], dtype=torch.long)

TOP_IDX = torch.tensor([4, 6, 8], dtype=torch.long)
BOT_IDX = torch.tensor([5, 7, 9], dtype=torch.long)


def fun_select(null_mat: torch.Tensor, i: int, j: int, ratio: int = 3) -> torch.Tensor:
    return null_mat[:, ratio * j + i]


def null_to_Nister_solution(X: torch.Tensor, batch_size: int) -> torch.Tensor:
    device = X.device
    dtype_internal = X.dtype

    T1 = T_deg1.to(device=device, dtype=dtype_internal)
    T2 = T_deg2.to(device=device, dtype=dtype_internal)
    mult_idx = multiplication_indices.to(device=device)
    sgns = signs.to(device=device, dtype=dtype_internal)
    coeff_map = coefficient_map.to(device=device)
    i_idx_dev = I_IDX.to(device=device)
    j_idx_dev = J_IDX.to(device=device)
    idx3_dev = IDX3.to(device=device)
    top_idx_dev = TOP_IDX.to(device=device)
    bot_idx_dev = BOT_IDX.to(device=device)
    idx_ij_dev = IDX_IJ.to(device=device)

    return _null_to_Nister_solution_script(
        X,
        batch_size,
        T1,
        T2,
        mult_idx,
        sgns,
        coeff_map,
        idx_ij_dev,
        i_idx_dev,
        j_idx_dev,
        idx3_dev,
        top_idx_dev,
        bot_idx_dev,
    )


def essential_from_fundamental(F_mat: torch.Tensor, K1: torch.Tensor, K2: torch.Tensor) -> torch.Tensor:
    r"""Get Essential matrix from Fundamental and Camera matrices.

    Uses the method from Hartley/Zisserman 9.6 pag 257 (formula 9.12).

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


def decompose_essential_matrix(E_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Decompose an essential matrix to possible rotations and translation.

    This function decomposes the essential matrix E using svd decomposition [96]
    and give the possible solutions: :math:`R1, R2, t`.

    Args:
       E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
       A tuple containing the first and second possible rotation matrices and the translation vector.
       The shape of the tensors with be same input :math:`[(*, 3, 3), (*, 3, 3), (*, 3, 1)]`.

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

    W = cross_product_matrix(torch.tensor([[0.0, 0.0, 1.0]]).type_as(E_mat))
    W[..., 2, 2] += 1.0

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

       Recover rotation and translation from essential matrices without SVD
      reference: Horn, Berthold KP. Recovering baseline and orientation from essential matrix[J].
      J. Opt. Soc. Am, 1990, 110.

    Args:
       E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
       A tuple containing the first and second possible rotation matrices and the translation vector.
       The shape of the tensors with be same input :math:`[(*, 3, 3), (*, 3, 3), (*, 3, 1)]`.

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
    R1 = (matrix_cofactor_tensor(E_mat) - B1 @ E_mat) / (b1 * b1).sum().unsqueeze(-1)
    R2 = (matrix_cofactor_tensor(E_mat) - B2 @ E_mat) / (b2 * b2).sum().unsqueeze(-1)

    return (R1, R2, b1_.unsqueeze(-1))


def essential_from_Rt(R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    r"""Get the Essential matrix from Camera motion (Rs and ts).

    Reference: Hartley/Zisserman 9.6 pag 257 (formula 9.12)

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
    3d points. Internally uses :py:meth:`~kornia.geometry.epipolar.decompose_essential_matrix` and then chooses
    the best solution based on the combination that gives more 3d points in front of the camera plane from
    :py:meth:`~kornia.geometry.epipolar.triangulate_points`.

    Args:
        E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.
        K1: The camera matrix from first camera with shape :math:`(*, 3, 3)`.
        K2: The camera matrix from second camera with shape :math:`(*, 3, 3)`.
        x1: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        x2: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        mask: A boolean mask which can be used to exclude some points from choosing
          the best solution. This is useful for using this function with sets of points of
          different cardinality (for instance after filtering with RANSAC) while keeping batch
          semantics. Mask is of shape :math:`(*, N)`.

    Returns:
        The rotation and translation plus the 3d triangulated points.
        The tuple is as following :math:`[(*, 3, 3), (*, 3, 1), (*, N, 3)]`.

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
    R_out = Rs[:, mask_indices][:, 0, 0]
    t_out = ts[:, mask_indices][:, 0, 0]
    points3d_out = X[:, mask_indices][:, 0, 0]

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

    Args:
         points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=5`.
         points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=5`.
         weights: Tensor containing the weights per point correspondence with a shape of :math:`(5, N)`.

    Returns:
         the computed essential matrices with shape :math:`(B, 10, 3, 3)`.
         Note that all possible solutions are returned, i.e., 10 essential matrices for each image pair.
         To choose the best one out of 10, try to check the one with the lowest Sampson distance.

    """
    E = run_5point(points1, points2, weights).to(points1.dtype)
    return E
