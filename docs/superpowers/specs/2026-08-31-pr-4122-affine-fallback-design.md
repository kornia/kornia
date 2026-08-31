# PR 4122 affine fallback design

## Goal

Finish PR 4122's half-precision support without turning a degenerate affine-shape estimate into a silent NaN in
either the forward or backward pass. Preserve each estimator's orientation contract, give the two interchangeable
affine estimators the same invalid-output policy, and restore the half-precision integration coverage that the PR
made reachable.

The closed-form inverse in `ellipse_to_laf` is already numerically pinned. Its optional tensor-assembly optimization
is outside this change.

## Design

### Prevent invalid values at their source

`PatchAffineShapeEstimator` will compare moments against `max(self.eps, torch.finfo(dtype).tiny)`. The default
`1e-10` remains unchanged for float32, float64, and bfloat16. In float16 it becomes the smallest normal value instead
of rounding to zero, so a zero or numerically degenerate moment matrix takes the existing circular-shape fallback.
This fixes the public patch estimator directly and reduces reliance on its callers' defenses.

### Sanitize before differentiable singular operations

`LAFAffineShapeEstimator` will classify an ellipse estimate as invalid when it contains non-finite values or either
diagonal coefficient is zero. It will replace only those shape coefficients with `[1, 0, 1]` before calling
`ellipse_to_laf`. Consequently, reciprocal and scale operations never see a singular estimate, so an inactive
`torch.where` branch cannot leak `0 * inf` into backward.

A private finalization helper will normalize a proposed LAF to the input LAF's scale and orientation. Before scale,
square-root, or orientation operations, it will replace invalid or zero-determinant proposals with a safe fallback.
Both `LAFAffineShapeEstimator` and `LAFAffNetShapeEstimator` will use this helper, eliminating their current policy
asymmetry.

The user-visible fallback is:

- the original LAF when `preserve_orientation=True`;
- `make_upright(laf)` when `preserve_orientation=False`.

The final selection remains a `torch.where`, but every value feeding its differentiable branches is finite. This
keeps the output exact on failed rows while keeping gradients finite for failed and neighboring valid keypoints.

### Test integration and documentation

Tests will be added before each production change and observed failing for the intended reason. Coverage will pin:

- finite image and LAF gradients for the float16 near-horizontal-edge reproduction;
- upright fallback for a rotated input when orientation preservation is disabled;
- dtype-aware circular fallback in `PatchAffineShapeEstimator`;
- the shared invalid-candidate behavior in `LAFAffNetShapeEstimator`;
- the existing padding contract in float16 and bfloat16 for both detector families.

The float16 regression will use the repository `dtype` fixture so CUDA half-precision skipping and subprocess
isolation apply. Runtime probes for convolution, grid sampling, and replicate padding will skip only combinations
whose PyTorch build lacks a required kernel. The stale `linalg.inv` skips will be removed, and comments and the
changelog will describe the resulting behavior rather than the superseded implementation detail.

## Compatibility and scope

Normal, finite estimates retain their current computation and outputs. The only behavioral changes are on invalid or
numerically degenerate estimates, which now produce a finite, contract-consistent fallback with finite backward, and
on direct float16 patch estimation, which now reaches its already-established circular fallback.

The design introduces no public API and no dependency. It does not add validation or synchronization to the public
`ellipse_to_laf` function, whose documented row-local non-finite behavior remains unchanged.

## Verification

Run the new tests through their red-green cycles, then run the focused affine-shape and scale-space detector tests in
float32/float64 and supported CPU half dtypes. Finish with `pixi run pre-commit-all`, `pixi run typecheck`, and the
relevant doctest suite. CUDA half tests must run only through `--isolate-half-precision` when CUDA is available.
