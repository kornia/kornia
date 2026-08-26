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
   reverse. Two "carefully chosen" sizes (257, 3001) each passed vacuously on kornia#4006.
   `unrepresentable_sizes` is `[]` for float32/float64 (every size below `2**24` is exact), so the
   bare sweep `sizes=unrepresentable_sizes(dtype)` would be an empty, vacuous test on those dtypes —
   which is why `assert_capture_matches_eager` raises `ValueError` on an empty `sizes`. Pass
   `sizes=[1, 2, *unrepresentable_sizes(dtype)[:8]]` instead.
2. **Under capture, divide by the unrounded size and cast the quotient.** Eager divides by a Python
   int that stays exact through float32 opmath; `(size_t - 1).to(half)` does not.
3. **Resolve `dtype=None` to `torch.get_default_dtype()` before deciding to promote.** The default
   dtype may itself be half. The parity sweep for any function taking `dtype=None` must include the
   cell `torch.set_default_dtype(<half>)` + `dtype=None`; explicit-`dtype=` scans cannot see this
   defect, and "scanned all four dtypes, unchanged" is not evidence for it. Shape:
   `previous = torch.get_default_dtype(); torch.set_default_dtype(torch.float16); try: ...
   finally: torch.set_default_dtype(previous)`.
4. **Guard a cast-back on `is_floating_point()`.** An integral coordinate dtype stays promoted, as
   eager's true-division leaves it.
5. **A degenerate path validates exactly what the full path validates.** Empty `dsize`, singleton
   axis, empty batch: same exception types for the same invalid input, checked with
   `assert_degenerate_path_parity`. A guard that is stricter or laxer on the empty path is a bug.
6. **Byte-equality between eager and capture**, `torch.equal` not `allclose` — the capture branch
   must perform the same rounding sequence.
7. **Compile tests carry `dynamo` or `compile` in the name** so `conftest.py` deselects them when
   `KORNIA_TEST_OPTIMIZER` is unset; use the `torch_optimizer` fixture. Trace tests do not need it.
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
assert_capture_matches_eager(fn, make_inputs, sizes=[1, 2, 258, 300], device=device,
                             dtype=torch.float32, capture="compile")
# the empty path must reject what the full path rejects
assert_degenerate_path_parity(warp_affine, full_kwargs, empty_kwargs, [("M", int_matrix), ("M", wrong_shape)])
```

Full reference: `TESTING.md` § "Precision and Degenerate-Path Helpers". Historical negatives that
these helpers must catch live in `tests/testing/_historical.py`.

Gate the PR with `pixi run verify-delta` (see `kornia-review-loop` step 6 for the flags and exit
codes) once these tests exist.

## Related

`kornia-review-loop` (the process this feeds), `kornia-developer` (compile-first changes).
