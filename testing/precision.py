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
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Literal, Sequence

import pytest
import torch

__all__ = ["assert_capture_matches_eager", "assert_degenerate_path_parity", "unrepresentable_sizes"]


def unrepresentable_sizes(dtype: torch.dtype, *, lo: int = 2, hi: int = 4096) -> list[int]:
    """Return the sizes in ``[lo, hi]`` at which ``n`` or ``n - 1`` is not exact in ``dtype``.

    A test of eager-vs-captured parity must sweep these rather than pick one: which operand a
    given implementation rounds is unknown in advance. In bfloat16, 257 rounds as a *size* but
    256 is an exact *divisor*, so a test at 257 passed vacuously against a divisor-rounding bug
    while catching a size-rounding one; 258 does the opposite.

    Args:
        dtype: a floating dtype. Integer dtypes raise ``TypeError``.
        lo: smallest size to consider (inclusive).
        hi: largest size to consider (inclusive). Must not exceed ``torch.finfo(dtype).max`` (65504
            for float16): above that candidates cast to non-finite values and converting them back
            to ``int64`` produces a sentinel rather than the original integer, so the result would
            be garbage rather than a usable size sweep.

    Returns:
        sorted sizes, empty when every integer in range is exact (float32/float64 below 2**24).

    Raises:
        TypeError: when ``dtype`` is not a floating dtype.
        ValueError: when ``hi`` exceeds ``torch.finfo(dtype).max``.

    Example:
        >>> unrepresentable_sizes(torch.bfloat16)[:3]
        [257, 258, 259]
        >>> unrepresentable_sizes(torch.float16)[:2]
        [2049, 2050]
    """
    if not dtype.is_floating_point:
        raise TypeError(f"unrepresentable_sizes needs a floating dtype, got {dtype}")
    finite_max = torch.finfo(dtype).max
    if hi > finite_max:
        raise ValueError(
            f"hi={hi} is above torch.finfo({dtype}).max={finite_max:g}: candidates there cast to non-finite "
            "values and conversion back to int64 yields a sentinel, so the sweep would be garbage rather than sizes"
        )
    n = torch.arange(lo, hi + 1, dtype=torch.int64)
    inexact_n = n.to(dtype).to(torch.int64) != n
    inexact_prev = (n - 1).to(dtype).to(torch.int64) != n - 1
    return n[inexact_n | inexact_prev].tolist()


@contextmanager
def _restoring_rng(device: torch.device) -> Iterator[None]:
    """Restore the RNG state on exit, so the next call inside starts where this one did.

    Tracing runs ``fn`` once itself, and the eager reference call runs it again; without this a
    function that consumes randomness would be compared against a different draw and fail with the
    rounding message below, which is not what went wrong.
    """
    cpu_state = torch.get_rng_state()
    module = getattr(torch, device.type, None) if device.type != "cpu" else None
    # ``torch.cuda``/``torch.mps``/``torch.xpu`` all take the device explicitly; the no-argument
    # form would read whichever device is current, which need not be the one under test.
    device_state = module.get_rng_state(device) if hasattr(module, "get_rng_state") else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if device_state is not None:
            module.set_rng_state(device_state, device)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Return ``t``'s raw bytes as a flat ``uint8`` tensor, for a bitwise rather than numeric compare.

    ``view(torch.uint8)`` needs a last stride of 1 and at least one dimension, so the tensor is made
    contiguous and flattened first; the callers have already established that the two shapes match.
    """
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def _as_tuple(out: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(out, torch.Tensor):
        return (out,)
    if isinstance(out, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in out):
        if not out:
            raise ValueError("assert_capture_matches_eager cannot compare an empty output sequence")
        return tuple(out)
    raise TypeError(f"assert_capture_matches_eager expects a tensor or a tuple of tensors, got {type(out)}")


def _snapshot(out: Any) -> tuple[torch.Tensor, ...]:
    """Copy outputs before a later invocation can mutate tensors that they alias."""
    return tuple(t.detach().clone() for t in _as_tuple(out))


def assert_capture_matches_eager(
    fn: Callable[..., Any],
    make_inputs: Callable[[int, torch.device, torch.dtype], tuple[torch.Tensor, ...]],
    *,
    sizes: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    capture: Literal["trace", "compile"] = "trace",
) -> None:
    """Assert that ``fn`` returns bit-identical outputs eagerly and under graph capture, per size.

    ``fn`` receives the tensors built by ``make_inputs(size, device, dtype)`` and must derive every
    size from a tensor shape (not from a Python int closed over), otherwise the capture cannot see
    the size at all. Outputs are compared as raw bytes -- the contract for a capture branch is the
    same rounding sequence as eager, not "close enough". That is deliberately stricter than
    ``torch.equal``, which is numeric rather than bitwise: it calls two NaNs unequal (a
    spurious failure) and ``+0.0`` equal to ``-0.0`` (a missed sign-bit change). Under a byte
    comparison a NaN matches a NaN with the same payload, and the two zeros differ.

    Args:
        fn: callable of the tensors produced by ``make_inputs``; returns a tensor or tuple of tensors.
        make_inputs: builds the inputs for one size; the size goes into a tensor shape.
        sizes: sizes to sweep. Write it as ``[1, 2, *unrepresentable_sizes(dtype)[:8]]``: the
            degenerate sizes 1 and 2 always exercise the singleton and smallest non-trivial
            paths, and they keep the list non-empty on an exact dtype, where
            ``unrepresentable_sizes`` is ``[]`` (float32/float64 below ``2**24``). Must not be
            empty -- a sweep of nothing is the vacuous green this helper exists to prevent.
        device: device for the inputs.
        dtype: dtype for the inputs (and the dtype the sizes were chosen for).
        capture: ``"trace"`` for ``torch.jit.trace`` (re-traced per size),
            ``"compile"`` for ``torch.compile(fullgraph=True, dynamic=True)``.

    Raises:
        ValueError: when ``capture`` is not ``"trace"`` or ``"compile"``, or when ``sizes`` is empty.
        AssertionError: on the first size where an output differs, naming size, output index, and
            either the shape/dtype mismatch, the max abs difference, or -- when the values agree but
            the bytes do not -- that only the bit pattern moved.

    """
    if capture not in ("trace", "compile"):
        raise ValueError(f"capture must be 'trace' or 'compile', got {capture!r}")
    if len(sizes) == 0:
        raise ValueError(
            "assert_capture_matches_eager needs at least one size: an empty sweep asserts nothing and "
            "passes silently. unrepresentable_sizes is empty for exact dtypes such as float32, so pass "
            "sizes=[1, 2, *unrepresentable_sizes(dtype)[:8]] rather than the bare call."
        )
    if capture == "compile" and torch.device(device).type == "mps":
        pytest.skip("torch.compile inductor backend is not available on MPS")
    if capture == "compile":
        # ``from torch import _dynamo`` rather than ``import torch._dynamo``: the latter would rebind
        # the name ``torch`` as a function-local and shadow the module-level import below.
        from torch import _dynamo

        _dynamo.reset()
        captured_fn = torch.compile(fn, fullgraph=True, dynamic=True)
    for size in sizes:
        if capture == "trace":
            with warnings.catch_warnings(), _restoring_rng(torch.device(device)):
                warnings.simplefilter("ignore", torch.jit.TracerWarning)
                trace_inputs = make_inputs(size, torch.device(device), dtype)
                captured_fn = torch.jit.trace(fn, trace_inputs, check_trace=False)
        with _restoring_rng(torch.device(device)):
            eager_inputs = make_inputs(size, torch.device(device), dtype)
            expected = _snapshot(fn(*eager_inputs))
        with _restoring_rng(torch.device(device)):
            captured_inputs = make_inputs(size, torch.device(device), dtype)
            actual = _snapshot(captured_fn(*captured_inputs))
        if len(expected) != len(actual):
            raise AssertionError(
                f"size {size}: eager returned {len(expected)} outputs, {capture} returned {len(actual)}"
            )
        for i, (e, a) in enumerate(zip(expected, actual)):
            if e.shape != a.shape or e.dtype != a.dtype:
                raise AssertionError(
                    f"size {size}, output {i}: eager {tuple(e.shape)} {e.dtype} vs {capture} {tuple(a.shape)} {a.dtype}"
                )
            if not torch.equal(_bits(e), _bits(a)):
                # via CPU float64: MPS has no float64, and subtracting in the tensor's own
                # half dtype would round the very difference being reported.
                lhs = e.detach().cpu().to(torch.float64)
                rhs = a.detach().cpu().to(torch.float64)
                # a NaN on both sides is the same *value*; only its payload bits moved
                if bool(((lhs == rhs) | (lhs.isnan() & rhs.isnan())).all()):
                    detail = "the values are equal but their bits are not (a signed zero or a NaN payload changed)"
                else:
                    # nan_to_num after the subtraction: a NaN on one side only must not hide the mismatch
                    diff = (lhs - rhs).abs().nan_to_num(nan=float("inf")).max().item()
                    detail = f"max abs diff {diff:.3g}"
                raise AssertionError(
                    f"size {size}, output {i}: {capture} output differs from eager, {detail} "
                    f"({e.dtype}). Under capture the size arithmetic must not round into {dtype}; "
                    "divide by the unrounded size and cast the quotient."
                )


def _outcome(fn: Callable[..., Any], kwargs: dict[str, Any]) -> type[BaseException] | None:
    try:
        fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 — the exception *type* is the observation
        return type(exc)
    return None


def assert_degenerate_path_parity(
    fn: Callable[..., Any],
    full_kwargs: dict[str, Any],
    degenerate_kwargs: dict[str, Any],
    bad_inputs: Sequence[tuple[str, Any]],
) -> None:
    """Assert that a degenerate (empty/singleton) path rejects exactly what the full path rejects.

    For each ``(name, bad_value)`` the function is called once with ``full_kwargs`` and once with
    ``degenerate_kwargs``, each with ``name`` replaced by ``bad_value``. Both calls must raise the
    same exception type, or both must succeed. A degenerate early-return that skips validation
    (laxer) or adds its own (stricter) fails here.

    Args:
        fn: the function under test, called with keyword arguments only.
        full_kwargs: a call that takes the ordinary, non-degenerate path and succeeds.
        degenerate_kwargs: the same call routed through the degenerate path (``dsize=(0, w)``, a
            singleton axis, an empty batch).
        bad_inputs: ``(argument name, invalid value)`` pairs to substitute into both calls. Each
            name must already be a key of both kwargs dicts. Must not be empty -- with no pairs
            only the two baseline calls run and nothing is compared.

    Raises:
        ValueError: when ``bad_inputs`` is empty, or a ``bad_inputs`` name is absent from either
            kwargs dict.
        AssertionError: naming the argument and the two outcomes, ``None`` meaning "no exception".

    Note:
        Parity is compared on the exception *type* only. Two paths that raise the same type for
        unrelated reasons read as parity here; check the messages when that is plausible.
    """
    if len(bad_inputs) == 0:
        raise ValueError(
            "assert_degenerate_path_parity needs at least one (name, bad_value) pair: with none, only the two "
            "baseline calls run, no invalid input is ever substituted, and the parity assertion passes having "
            "compared nothing -- the vacuous green this helper exists to prevent."
        )
    for name, _ in bad_inputs:
        # substituting an unknown name *adds* a key rather than replacing one, so both paths raise
        # TypeError("unexpected keyword argument") and the assertion below passes having tested nothing
        missing = [
            label for label, kwargs in (("full", full_kwargs), ("degenerate", degenerate_kwargs)) if name not in kwargs
        ]
        if missing:
            raise ValueError(
                f"{name!r} is not a key of {' or '.join(missing)}_kwargs; substituting an unknown argument name "
                "raises TypeError on both paths and passes vacuously"
            )
    for label, kwargs in (("full", full_kwargs), ("degenerate", degenerate_kwargs)):
        baseline = _outcome(fn, kwargs)
        if baseline is not None:
            raise AssertionError(f"the {label} baseline call raised {baseline.__name__}; fix the fixture first")
    for name, bad in bad_inputs:
        full = _outcome(fn, {**full_kwargs, name: bad})
        degenerate = _outcome(fn, {**degenerate_kwargs, name: bad})
        if full is not degenerate:
            raise AssertionError(
                f"{name}={type(bad).__name__}: full path -> {full and full.__name__} vs degenerate path -> "
                f"{degenerate and degenerate.__name__}. The degenerate path must validate exactly what the "
                "full path validates."
            )
