## What does this pull request do?

Adds mutation pins for all three filter changes reported in #5509: the upper and lower simple-root derivative threshold mutations, and tightening the recovered-placeholder step bound from `sqrt(eps)` to `eps`. The float16 coefficients represent `(x + 1.5)^2 (x + 2.5) (x + 3)` and verify both double-root copies survive. A separate float32 quartic verifies that a recovered small root is retained. The float32 retry masks the reported upper-threshold example, so its calibrated value is asserted explicitly.

Adds the fixed changelog fragment `changelog.d/5621.fixed.md`.

## Related issue or discussion

Fixes #5509

## What did you check?

```text
$ .venv/bin/pytest tests/geometry/solvers/test_polynomial_solver.py -k simple_root_threshold_for_half_inputs_5509 --device=cpu --dtype=float16
1 passed

$ .venv/bin/pytest tests/geometry/solvers/test_polynomial_solver.py -k 'not test_random' --device=cpu --dtype=float16,float32,float64
441 passed, 104 skipped, 3 deselected.

$ PATH=/usr/bin:/opt/homebrew/bin:$PATH pixi run -e default pre-commit-all
Passed

$ PATH=/usr/bin:/opt/homebrew/bin:$PATH pixi run -e default uv build --no-sources
Built source distribution and wheel.
```

The full solver test module also has an existing float16 `test_random` failure. The same test fails on the unchanged base revision with the same residual mismatch, so it is unrelated to this test-only change.

## How did AI help?

AI helped investigate the mutation case and draft the regression test. The test behavior was reproduced directly, and the unrelated float16 failure was compared against the base revision.
