---
name: kornia-precision-testing
description: Use when writing or reviewing a kornia test that involves float16/bfloat16, device, torch.jit.trace / torch.compile, or degenerate sizes (0, 1, empty dsize). Picks sizes that cannot pass vacuously and uses the testing.precision helpers so eager-vs-capture and full-vs-degenerate parity are asserted by code, not by a hand-picked "boundary" case.
---

# kornia-precision-testing

Three helpers in `testing/precision.py` enforce the rules below. Prefer them over hand-written
sweeps; when a rule cannot be expressed through a helper, cite the rule in the test's comment.

## Rules

1. **Sweep `unrepresentable_sizes(dtype)`, never pick one size.** A size is dangerous when `n` OR
   `n - 1` is inexact in the dtype, and which one matters depends on the implementation: bfloat16
   257 caught a size-rounding bug and passed vacuously against a divisor-rounding bug; 258 does the
   reverse. Two "carefully chosen" sizes (257, 3001) each passed vacuously on kornia#4006. Always
   pass `sizes=[1, 2, *unrepresentable_sizes(dtype)[:8]]`, never the bare call: the list is `[]` for
   float32/float64 (every size below `2**24` is exact), and `assert_capture_matches_eager` refuses
   an empty sweep rather than passing silently. The helper enforces its own bounds — a non-floating
   dtype and an `hi` above `torch.finfo(dtype).max` both raise. Float16's first dangerous sizes
   start at 2049, so a 2-D+ workload can be expensive; keep the required sweep, but use the
   smallest representative prefix and a narrowly shaped input rather than silently replacing it
   with 257 or another hand-picked size.
2. **Under capture, divide by the unrounded size and cast the quotient.** Eager divides by a Python
   int that stays exact through float32 opmath; `(size_t - 1).to(half)` does not.
3. **Resolve `dtype=None` to `torch.get_default_dtype()` before deciding to promote.** The default
   dtype may itself be half. The parity sweep for any function taking `dtype=None` must include the
   cell `torch.set_default_dtype(<half>)` + `dtype=None`; explicit-`dtype=` scans cannot see this
   defect, and "scanned all four dtypes, unchanged" is not evidence for it. Shape (see
   `tests/geometry/test_conversions.py`'s `_ambient_default_dtype` helper for the same pattern):

   ```python
   previous = torch.get_default_dtype()
   torch.set_default_dtype(torch.float16)
   try:
       ...  # the dtype=None cell under test
   finally:
       torch.set_default_dtype(previous)
   ```

   `set_default_dtype` is a process-global mutation, so the restore must sit in a `finally` that
   the assertion cannot jump over, inside the test itself. Do not hand-roll save/restore in a
   yielding fixture: a failure in a later fixture, or a `KeyboardInterrupt`, can leave the whole
   session in half precision. A fixture that yields a properly scoped context manager is acceptable
   when the test enters that context itself, so its `finally` encloses the assertion.

4. **Guard a cast-back on `is_floating_point()`.** An integral coordinate dtype stays promoted, as
   eager's true-division leaves it.
5. **A degenerate path validates exactly what the full path validates.** Empty `dsize`, singleton
   axis, empty batch: same exception types for the same invalid input, checked with
   `assert_degenerate_path_parity`. When the same exception class has plausible distinct sources,
   also disambiguate the cause/message (and, when useful, the first library traceback frame): a
   matching `ValueError` from an unrelated shape check is not parity. A guard that is stricter or
   laxer on the empty path is a bug.
   Every `bad_inputs` name must be a real argument already present in both kwargs dicts — the
   helper raises rather than let a typo'd name be *added* to the call, where both paths raise
   `TypeError: unexpected keyword argument` and parity holds over a call that never happened. It
   also refuses an empty `bad_inputs`, which would run the two baseline calls and compare nothing.
6. **Byte-equality between eager and capture**, not `allclose` — the capture branch must perform
   the same rounding sequence. `assert_capture_matches_eager` compares the raw bytes rather than
   `torch.equal`, which is numeric: it reads two NaNs as unequal (a spurious failure on any op that
   legitimately produces one) and `+0.0` as equal to `-0.0` (a sign-bit change it would miss).
7. **Compile tests carry `dynamo` or `compile` in the name** so `conftest.py` deselects them when
   `KORNIA_TEST_OPTIMIZER` is unset. `capture="compile"` hardcodes `torch.compile`; it does not
   honor the `torch_optimizer` fixture. Route optimizer-specific compile coverage through a
   separate test using that fixture. Trace tests do not need it.
8. **MPS and half-precision skips copy the nearest existing test's skip**, with the reason. Do not
   invent a new tolerance; `BaseTester.assert_close` already has dtype-specific ones.
9. **A regression test must fail on the pre-fix SHA.** Check it before trusting it.
10. **Historical negatives in `tests/testing/_historical.py` are trace-only fixtures** — do not run
    them under `capture="compile"`.
11. **A `torch.compiler.is_compiling()` branch needs a `capture="compile"` sweep, not just
    `"trace"`.** The inductor `-k "dynamo or compile"` surface only runs tests that exist, so a
    compile-only divergence in a new code path has no other net.

## Helpers

```python
from testing import assert_capture_matches_eager, assert_degenerate_path_parity, unrepresentable_sizes

# eager vs trace, sizes derived from a tensor shape inside fn
assert_capture_matches_eager(fn, make_inputs, sizes=[1, 2, *unrepresentable_sizes(dtype)[:8]],
                             device=device, dtype=dtype, capture="trace")
# eager vs torch.compile(fullgraph=True, dynamic=True); name the test test_*_compile_*
# Parameterize dtype over relevant floating dtypes; do not substitute hand-picked sizes.
assert_capture_matches_eager(fn, make_inputs, sizes=[1, 2, *unrepresentable_sizes(dtype)[:8]],
                             device=device, dtype=dtype, capture="compile")
# the empty path must reject what the full path rejects
assert_degenerate_path_parity(warp_affine, full_kwargs, empty_kwargs, [("M", int_matrix), ("M", wrong_shape)])
```

Full reference: `TESTING.md` § "Precision and Degenerate-Path Helpers". Historical negatives that
these helpers must catch live in `tests/testing/_historical.py`.

Gate the PR with `pixi run verify-delta` (see `kornia-review-loop` step 6 for the flags, the exit
codes, and why exit 0 alone is not the verdict) once these tests exist.

## Related

`kornia-review-loop` (the process this feeds), `kornia-developer` (compile-first changes).
