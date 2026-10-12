Fixes #5773

## What does this pull request do?

Use reduced SVD factors in `solve_pnp_dlt` and the `run_5point` least-squares path. Both tall-matrix paths use only the right singular vectors; reduced SVD preserves those factors while avoiding the unused square left-singular matrix and its backward storage.

The `run_5point` path keeps the existing full-SVD null-space implementation when fewer than nine correspondences are supplied, where the full right-singular basis is required.

## What did you check?

- The `test_pnp.py` and `test_essential.py` suites: 238 passed, 2 skipped on CPU with float32 and float64 using Torch 2.5.1 and 2.14.
- Added autograd saved-tensor storage regressions for both solvers; they also verify finite gradients.
- Changed-file pre-commit hooks passed.
- Built the Kornia wheel successfully with `uv build --wheel`.
