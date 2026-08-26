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

import warnings
from typing import Any, Callable, Literal, Sequence

import pytest
import torch

__all__ = ["assert_capture_matches_eager", "unrepresentable_sizes"]


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


def _as_tuple(out: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(out, torch.Tensor):
        return (out,)
    if isinstance(out, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in out):
        return tuple(out)
    raise TypeError(f"assert_capture_matches_eager expects a tensor or a tuple of tensors, got {type(out)}")


def assert_capture_matches_eager(
    fn: Callable[..., Any],
    make_inputs: Callable[[int, torch.device, torch.dtype], tuple[torch.Tensor, ...]],
    *,
    sizes: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    capture: Literal["trace", "compile"] = "trace",
) -> None:
    """Assert that ``fn`` returns byte-identical outputs eagerly and under graph capture, per size.

    ``fn`` receives the tensors built by ``make_inputs(size, device, dtype)`` and must derive every
    size from a tensor shape (not from a Python int closed over), otherwise the capture cannot see
    the size at all. Comparison is ``torch.equal`` -- the contract for a capture branch is the same
    rounding sequence as eager, not "close enough".

    Args:
        fn: callable of the tensors produced by ``make_inputs``; returns a tensor or tuple of tensors.
        make_inputs: builds the inputs for one size; the size goes into a tensor shape.
        sizes: sizes to sweep -- normally ``unrepresentable_sizes(dtype)`` or a slice of it, plus
            the degenerate sizes 1 and 2.
        device: device for the inputs.
        dtype: dtype for the inputs (and the dtype the sizes were chosen for).
        capture: ``"trace"`` for ``torch.jit.trace`` (re-traced per size),
            ``"compile"`` for ``torch.compile(fullgraph=True, dynamic=True)``.

    Raises:
        AssertionError: on the first size where an output differs, naming size, output index,
            shape/dtype mismatch or max abs difference.

    """
    if capture == "compile" and torch.device(device).type == "mps":
        pytest.skip("torch.compile inductor backend is not available on MPS")
    if capture == "compile":
        # ``from torch import _dynamo`` rather than ``import torch._dynamo``: the latter would rebind
        # the name ``torch`` as a function-local and shadow the module-level import below.
        from torch import _dynamo

        _dynamo.reset()
        captured_fn = torch.compile(fn, fullgraph=True, dynamic=True)
    for size in sizes:
        inputs = make_inputs(size, torch.device(device), dtype)
        if capture == "trace":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", torch.jit.TracerWarning)
                captured_fn = torch.jit.trace(fn, inputs, check_trace=False)
        expected = _as_tuple(fn(*inputs))
        actual = _as_tuple(captured_fn(*inputs))
        if len(expected) != len(actual):
            raise AssertionError(
                f"size {size}: eager returned {len(expected)} outputs, {capture} returned {len(actual)}"
            )
        for i, (e, a) in enumerate(zip(expected, actual)):
            if e.shape != a.shape or e.dtype != a.dtype:
                raise AssertionError(
                    f"size {size}, output {i}: eager {tuple(e.shape)} {e.dtype} vs {capture} {tuple(a.shape)} {a.dtype}"
                )
            if not torch.equal(e, a):
                # via CPU float64: MPS has no float64, and subtracting in the tensor's own
                # half dtype would round the very difference being reported.
                lhs = e.detach().cpu().to(torch.float64)
                rhs = a.detach().cpu().to(torch.float64)
                diff = (lhs - rhs).abs().max().item()
                raise AssertionError(
                    f"size {size}, output {i}: {capture} output differs from eager, max abs diff {diff:.3g} "
                    f"({e.dtype}). Under capture the size arithmetic must not round into {dtype}; "
                    "divide by the unrounded size and cast the quotient."
                )
