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

"""Helpers for tests that exercise reduced-precision dtypes, graph capture, and degenerate sizes.

These exist because the same three bug classes recurred across kornia#4006's review rounds:
a size rounded into a half dtype before a division under ``torch.jit.trace``/``torch.compile``,
regression tests that passed vacuously because the chosen size happened to be exact, and an
empty/singleton code path that validated less (or more) than the non-empty path.
"""

from __future__ import annotations

import torch

__all__ = ["unrepresentable_sizes"]


def unrepresentable_sizes(dtype: torch.dtype, *, lo: int = 2, hi: int = 4096) -> list[int]:
    """Return the sizes in ``[lo, hi]`` at which ``n`` or ``n - 1`` is not exact in ``dtype``.

    A test of eager-vs-captured parity must sweep these rather than pick one: which operand a
    given implementation rounds is unknown in advance. In bfloat16, 257 rounds as a *size* but
    256 is an exact *divisor*, so a test at 257 passed vacuously against a divisor-rounding bug
    while catching a size-rounding one; 258 does the opposite.

    Args:
        dtype: a floating dtype. Integer dtypes raise ``TypeError``.
        lo: smallest size to consider (inclusive).
        hi: largest size to consider (inclusive).

    Returns:
        sorted sizes, empty when every integer in range is exact (float32/float64 below 2**24).

    Example:
        >>> unrepresentable_sizes(torch.bfloat16)[:3]
        [257, 258, 259]
        >>> unrepresentable_sizes(torch.float16)[:2]
        [2049, 2050]
    """
    if not dtype.is_floating_point:
        raise TypeError(f"unrepresentable_sizes needs a floating dtype, got {dtype}")
    n = torch.arange(lo, hi + 1, dtype=torch.int64)
    inexact_n = n.to(dtype).to(torch.int64) != n
    inexact_prev = (n - 1).to(dtype).to(torch.int64) != n - 1
    return n[inexact_n | inexact_prev].tolist()
