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

"""Closed-form solvers for homogeneous linear systems."""

from __future__ import annotations

import torch

from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE


def _det3(
    a0: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
) -> torch.Tensor:
    """Compute a batch of 3x3 determinants via Sarrus' rule.

    Given three rows (each split into their three scalar components), returns::

        | a0  a1  a2 |
        | b0  b1  b2 |
        | c0  c1  c2 |

    All inputs must be broadcastable to the same shape.

    Args:
        a0: first element of the first row.
        a1: second element of the first row.
        a2: third element of the first row.
        b0: first element of the second row.
        b1: second element of the second row.
        b2: third element of the second row.
        c0: first element of the third row.
        c1: second element of the third row.
        c2: third element of the third row.

    Returns:
        Scalar determinant (or batch of scalars) with the broadcasted shape.
    """
    return a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0)


def null_vector_3x4(A: torch.Tensor) -> torch.Tensor:
    r"""Return the null vector of a rank-3 matrix of shape :math:`(*, 3, 4)`.

    The null vector :math:`\mathbf{v} \in \mathbb{R}^4` satisfies
    :math:`A\,\mathbf{v} = \mathbf{0}`.  For a matrix of rank 3 this solution
    is unique up to scale.

    The computation uses the **4-D cross-product** (cofactor expansion):
    each component of :math:`\mathbf{v}` is a :math:`3 \times 3` determinant of
    the submatrix obtained by dropping the corresponding column of :math:`A`.
    For a rank-3 :math:`A` this gives the last right singular vector up to scale
    and sign, but replaces the SVD with 48 scalar multiplications and 20
    additions — no LAPACK or cuSOLVER call is made. The sign follows the
    cofactor formula below (``[I | 0]`` gives ``[0, 0, 0, -1]``), and a lower
    rank gives the zero vector rather than a unit vector.

    .. math::

        v_j = (-1)^j \det\!\bigl(A_{[0,1,2],\,\widehat{j}}\bigr), \quad j = 0, 1, 2, 3

    where :math:`A_{[0,1,2],\,\widehat{j}}` denotes the :math:`3 \times 3`
    submatrix formed by deleting column :math:`j`.

    .. note::

        The returned vector is **not** normalised.  Divide by its norm if a
        unit null vector is required.

    .. note::

        The function is only correct when :math:`A` has rank exactly 3.  For
        rank-deficient inputs (rank < 3) the result is the zero vector.

    Args:
        A: matrix of shape :math:`(*, 3, 4)`.

    Returns:
        Null vector of shape :math:`(*, 4)`.

    Raises:
        TypeCheckError: if ``A`` is not a :class:`torch.Tensor`.
        ShapeError: if the last two dimensions of ``A`` are not ``(3, 4)``.

    Example:
        >>> A = torch.tensor([[[1., 0., 0., 0.],
        ...                    [0., 1., 0., 0.],
        ...                    [0., 0., 1., 0.]]])   # null vector is [0,0,0,-1]
        >>> v = null_vector_3x4(A)                   # shape (1, 4)
        >>> (A @ v.unsqueeze(-1)).squeeze(-1)         # should be near zero
        tensor([[0., 0., 0.]])

    """
    KORNIA_CHECK_IS_TENSOR(A)
    KORNIA_CHECK_SHAPE(A, ["*", "3", "4"])

    a = A[..., 0, :]  # (*, 4)
    b = A[..., 1, :]  # (*, 4)
    c = A[..., 2, :]  # (*, 4)

    # Each component of the null vector is a signed 3x3 cofactor determinant.
    v0 = _det3(
        a[..., 1],
        a[..., 2],
        a[..., 3],
        b[..., 1],
        b[..., 2],
        b[..., 3],
        c[..., 1],
        c[..., 2],
        c[..., 3],
    )
    v1 = -_det3(
        a[..., 0],
        a[..., 2],
        a[..., 3],
        b[..., 0],
        b[..., 2],
        b[..., 3],
        c[..., 0],
        c[..., 2],
        c[..., 3],
    )
    v2 = _det3(
        a[..., 0],
        a[..., 1],
        a[..., 3],
        b[..., 0],
        b[..., 1],
        b[..., 3],
        c[..., 0],
        c[..., 1],
        c[..., 3],
    )
    v3 = -_det3(
        a[..., 0],
        a[..., 1],
        a[..., 2],
        b[..., 0],
        b[..., 1],
        b[..., 2],
        c[..., 0],
        c[..., 1],
        c[..., 2],
    )

    return torch.stack([v0, v1, v2, v3], dim=-1)


def _null_space_lu(A: torch.Tensor) -> torch.Tensor:
    r"""Right null spaces of a batch of ``(B, m, n)`` matrices, ``m < n``, as ``(B, n, n - m)``.

    With ``A^T = P L U`` from a partial-pivoted LU factorization, ``f^T A^T = 0`` exactly when ``y = P^T f`` solves
    ``y^T L = 0``. Splitting the unit lower trapezoidal ``L`` into its square top ``L_1`` and bottom ``L_2`` rows
    gives the basis ``y = [-(L_2 L_1^{-1})^T; I]``. The pivoting chooses the gauge, so no coordinate of the null
    vector is assumed non-zero, and the basis is not orthonormal.

    Unlike an SVD, a QR or an ``eigh`` of ``A^T A``, a batched LU factorization is one batched kernel on every
    backend, and working on ``A`` rather than ``A^T A`` does not square its condition number. ``L_1`` is unit
    triangular, so a rank-deficient ``A`` still gives a basis of null vectors, of dimension ``n - m`` only, as long
    as the factorization stays finite; callers treat non-finite vectors as degenerate.
    """
    batch, m, n = A.shape
    lu, pivots, _ = torch.linalg.lu_factor_ex(A.mT)
    square = lu[:, :m, :m]
    if torch.compiler.is_compiling() and A.device.type == "cuda":
        # The CUDA meta kernel tests column-major contiguity of this strided LU view. With an unbacked
        # batch inside while_loop that needs a data-dependent guard. Make that layout check always true,
        # including empty batches; a row-major copy still needs a guard for the empty case.
        square = square.mT.contiguous().mT
    rhs = lu[:, m:, :m]
    if A.device.type == "mps":
        # MPS solve_triangular reads strided views wrongly on torch 2.5.1 and 2.9.1.
        square, rhs = square.contiguous(), rhs.contiguous()
    lower = torch.linalg.solve_triangular(square, rhs, upper=False, left=False, unitriangular=True)
    eye = torch.eye(n - m, dtype=A.dtype, device=A.device).expand(batch, -1, -1)
    permutation, _, _ = torch.lu_unpack(lu, pivots, unpack_data=False)
    return permutation @ torch.cat([-lower.mT, eye], 1)


def _null_space_householder(A: torch.Tensor) -> torch.Tensor:
    r"""Orthonormal right null spaces of a batch of ``(B, m, n)`` matrices, ``m < n``, as ``(B, n, n - m)``.

    The last ``n - m`` columns of ``Q`` in the QR factorization ``A^T = Q R``, from ``m`` Householder reflections
    written as batched tensor operations: ``torch.linalg.qr`` has no batched CUDA kernel and loops over the batch.
    Both this and :func:`_null_space_lu` span the null space to rounding, but a solver that parametrizes its solution
    in the basis can depend on which basis it gets: on 22000 exact five-point samples, Nister's candidates missed the
    true essential matrix by more than 1e-3 nine times with this orthonormal basis, seven times with the SVD's, and 81
    times with the LU one. It costs about four times the LU null space. ``A`` must have full rank: at a vanishing
    reflector the normalization divides by zero.
    """
    batch, m, n = A.shape
    remaining = A.mT  # (B, n, m): the columns still to be reduced, below the rows already done
    reflectors = []
    for _ in range(m):
        x = remaining[:, :, 0]
        head = x[:, :1]
        # The sign that avoids cancellation in the first entry of the reflector.
        sign = torch.where(head >= 0, torch.ones_like(head), -torch.ones_like(head))
        v = torch.cat([head + sign * x.norm(dim=1, keepdim=True), x[:, 1:]], 1)
        v = v / v.norm(dim=1, keepdim=True)
        reflectors.append(v)
        remaining = (remaining - 2 * v[:, :, None] * (v[:, None, :] @ remaining))[:, 1:, 1:]
    # Q e_j for j >= m, with Q = H_0 H_1 ... H_{m-1}: the reflectors in reverse order, H_k acting on rows k and below.
    Q = torch.eye(n, dtype=A.dtype, device=A.device)[:, m:].expand(batch, n, n - m)
    for k in reversed(range(m)):
        v, tail = reflectors[k], Q[:, k:]
        Q = torch.cat([Q[:, :k], tail - 2 * v[:, :, None] * (v[:, None, :] @ tail)], 1)
    return Q
