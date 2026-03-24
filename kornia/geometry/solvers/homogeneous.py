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

from kornia.core.check import KORNIA_CHECK_SHAPE


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
    This is equivalent to computing the last right singular vector of :math:`A`
    via SVD but replaces the SVD with 48 scalar multiplications and 20
    additions — no LAPACK or cuSOLVER call is made.

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
        TypeError: if ``A`` is not a tensor.
        ValueError: if the last two dimensions of ``A`` are not ``(3, 4)``.

    Example:
        >>> A = torch.tensor([[[1., 0., 0., 0.],
        ...                    [0., 1., 0., 0.],
        ...                    [0., 0., 1., 0.]]])   # null vector is [0,0,0,1]
        >>> v = null_vector_3x4(A)                   # shape (1, 4)
        >>> (A @ v.unsqueeze(-1)).squeeze(-1)         # should be near zero
        tensor([[0., 0., 0.]])

    """
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
