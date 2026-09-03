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

"""Closed-form arithmetic for 2x2, 3x3 and 4x4 matrices.

Private. These are **kernels**, not an API: every one of them is basic arithmetic on a
fixed-size matrix, with no LAPACK, cuSOLVER or backend ``linalg`` call. That is what makes
them usable where ``torch.linalg`` is not -- neither ONNX exporter lowers
``aten::linalg_inv``, the Jetson wheel fails to ``dlopen`` its LAPACK backend, MPS has no
kernel for several decompositions, and the ``linalg`` family has no half-precision kernel on
any backend -- on CUDA ``linalg.inv`` gives ``Low precision dtypes not supported. Got Half``
and ``linalg.det`` gives ``"lu_factor_cusolver" not implemented for 'Half'``.

The contract is deliberately minimal, because the callers own the policy:

- They compute in whatever dtype the caller supplies. **No promotion** -- that belongs to
  :func:`kornia.core.utils._torch_inverse_cast`.
- Exact shape and a real floating dtype are **caller preconditions, not runtime checks**.
  Behavior on anything else is unspecified: ``_adjugate_3x3`` of a 4x4 silently uses the
  leading 3x3 block, and of a 2x2 raises an incidental ``IndexError``. The contractual size
  guard lives on the dispatcher :func:`kornia.core.utils._adjugate_closed_form`.
- Output for a singular input is unspecified. There is no validity or conditioning
  guarantee; a zero determinant is not a reliable singularity test in floating point.
- No cross-mode bit-identity promise.

Execution-mode policy (which kernel a caller takes under tracing or export), size dispatch,
dtype promotion and validation all live in :mod:`kornia.core.utils`. Keeping them there is
what lets this module import nothing from kornia and stay a leaf.
"""

from typing import Tuple

import torch

__all__ = [
    "_adjugate_2x2",
    "_adjugate_3x3",
    "_adjugate_4x4",
    "_inverse_3x3_cross",
    "_inverse_3x3_scalar",
]


def _adjugate_2x2(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the adjugate and determinant of batched 2x2 matrices, in basic arithmetic only."""
    a = input[..., 0, 0]
    b = input[..., 0, 1]
    c = input[..., 1, 0]
    d = input[..., 1, 1]
    det = a * d - b * c
    adj = torch.stack([torch.stack([d, -b], dim=-1), torch.stack([-c, a], dim=-1)], dim=-2)
    return adj, det


def _adjugate_3x3(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the adjugate and determinant of batched 3x3 matrices, in basic arithmetic only."""
    a = input[..., 0, 0]
    b = input[..., 0, 1]
    c = input[..., 0, 2]
    d = input[..., 1, 0]
    e = input[..., 1, 1]
    f = input[..., 1, 2]
    g = input[..., 2, 0]
    h = input[..., 2, 1]
    i = input[..., 2, 2]

    # Cofactors (signed minors).
    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    c10 = -(b * i - c * h)
    c11 = a * i - c * g
    c12 = -(a * h - b * g)
    c20 = b * f - c * e
    c21 = -(a * f - c * d)
    c22 = a * e - b * d

    det = a * c00 + b * c01 + c * c02

    # Adjugate is the transpose of the cofactor matrix.
    row0 = torch.stack([c00, c10, c20], dim=-1)
    row1 = torch.stack([c01, c11, c21], dim=-1)
    row2 = torch.stack([c02, c12, c22], dim=-1)
    adj = torch.stack([row0, row1, row2], dim=-2)

    return adj, det


def _adjugate_4x4(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the adjugate and determinant of batched 4x4 matrices, in basic arithmetic only.

    Laplace expansion over the 2x2 minors of the top two rows (``s``) and bottom two rows (``c``),
    the standard 4x4 cofactor scheme (e.g. MESA ``gluInvertMatrix``).
    """
    a = input[..., 0, 0]
    b = input[..., 0, 1]
    c = input[..., 0, 2]
    d = input[..., 0, 3]
    e = input[..., 1, 0]
    f = input[..., 1, 1]
    g = input[..., 1, 2]
    h = input[..., 1, 3]
    i = input[..., 2, 0]
    j = input[..., 2, 1]
    k = input[..., 2, 2]
    l_ = input[..., 2, 3]
    m = input[..., 3, 0]
    n = input[..., 3, 1]
    o = input[..., 3, 2]
    p = input[..., 3, 3]

    s0 = a * f - b * e
    s1 = a * g - c * e
    s2 = a * h - d * e
    s3 = b * g - c * f
    s4 = b * h - d * f
    s5 = c * h - d * g

    c5 = k * p - l_ * o
    c4 = j * p - l_ * n
    c3 = j * o - k * n
    c2 = i * p - l_ * m
    c1 = i * o - k * m
    c0 = i * n - j * m

    det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0

    row0 = torch.stack(
        [f * c5 - g * c4 + h * c3, -b * c5 + c * c4 - d * c3, n * s5 - o * s4 + p * s3, -j * s5 + k * s4 - l_ * s3],
        dim=-1,
    )
    row1 = torch.stack(
        [-e * c5 + g * c2 - h * c1, a * c5 - c * c2 + d * c1, -m * s5 + o * s2 - p * s1, i * s5 - k * s2 + l_ * s1],
        dim=-1,
    )
    row2 = torch.stack(
        [e * c4 - f * c2 + h * c0, -a * c4 + b * c2 - d * c0, m * s4 - n * s2 + p * s0, -i * s4 + j * s2 - l_ * s0],
        dim=-1,
    )
    row3 = torch.stack(
        [-e * c3 + f * c1 - g * c0, a * c3 - b * c1 + c * c0, -m * s3 + n * s1 - o * s0, i * s3 - j * s1 + k * s0],
        dim=-1,
    )
    adj = torch.stack([row0, row1, row2, row3], dim=-2)

    return adj, det


def _inverse_3x3_cross(input: torch.Tensor) -> torch.Tensor:
    """Closed-form 3x3 inverse via three fused cross products. The eager path.

    ``inv(M) = adj(M) / det``, and for a 3x3 the adjugate rows are cross products of the
    columns: with columns ``(a, b, c)``, the inverse rows are ``(b x c, c x a, a x b) / det``,
    ``det = a . (b x c)``. Three fused ``cross`` ops instead of nine scalar cofactor
    expressions and four stacks -- far fewer kernel launches, which dominate on small matrices.

    Prefer :func:`_inverse_3x3_scalar` wherever a graph is being captured. The reason is
    **kernel coverage, not ONNX lowering**: a backend with no ``cross`` kernel for the working
    dtype makes this raise rather than return -- torch 2.5.1 has no ``bfloat16`` ``cross`` on
    MPS, while 2.9.1 does.

    ONNX lowering is *not* a reason, measured: ``torch.linalg.cross`` lowers on both torch
    versions kornia's CI runs -- 2.5.1 (legacy exporter) and 2.9.1 (legacy and dynamo) -- to
    ``Slice``/``Mul``/``Sub``/``Concat``. What neither exporter lowers on either version is
    ``aten::linalg_inv``, which is what :func:`kornia.core.utils._torch_inverse_cast` avoids by
    reaching for a closed form in the first place. torch 2.0-2.4 is untested; CI does not run it
    either.
    """
    col_a = input[..., :, 0]
    col_b = input[..., :, 1]
    col_c = input[..., :, 2]
    row0 = torch.linalg.cross(col_b, col_c, dim=-1)
    row1 = torch.linalg.cross(col_c, col_a, dim=-1)
    row2 = torch.linalg.cross(col_a, col_b, dim=-1)
    det = (col_a * row0).sum(-1)
    return torch.stack([row0, row1, row2], dim=-2) / det[..., None, None]


def _inverse_3x3_scalar(input: torch.Tensor) -> torch.Tensor:
    """Closed-form 3x3 inverse via the plain scalar adjugate. The trace/export path.

    Lowers to basic arithmetic that every standard ONNX opset supports, and calls no ``cross``.
    Algebraically identical to :func:`_inverse_3x3_cross`; the two differ only in the order the
    same products are accumulated, so they agree to rounding rather than bit-for-bit.
    """
    adj, det = _adjugate_3x3(input)
    return adj / det[..., None, None]
