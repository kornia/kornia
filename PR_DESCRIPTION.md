## What does this pull request do?

Restores `torch.export` and full-graph compilation for the default Otsu path, preserves the histogram
foreground when a floating threshold hits a pixel, and chooses the earliest equivalent split across empty bins.

- Computes all default histograms with batched tensor operations, using the same bin-assignment arithmetic as
  `torch.histc`. Constant planes keep their exact input value, including large integers.
- Limits default floating thresholds to the interval between the selected background and foreground pixels.
  This handles exact edges, half-precision rounding, and fused interpolation during compilation. The slow
  path receives only the collision correction, retaining its existing histogram calculation.
- Excludes empty bins from default split candidates, preventing MPS cumulative-sum rounding from selecting
  a later edge with the same partition. This intentionally changes affected returned thresholds and has a
  breaking changelog fragment.
- Uses a portable half-precision predecessor and accounts for MPS float32/bfloat16 comparisons flushing
  subnormal values at zero. The public mask remains exactly `input > threshold`.
- Adds regression tests and refreshes the three default-path graph-capture support entries using the
  repository survey harness on PyTorch 2.9.1, ONNX 1.21.0, ONNX Runtime 1.23.2, and ONNX Script 0.5.7.

No public API names, signatures, or runtime dependencies change.

## Related issue or discussion

Fixes #5425
Fixes #5422
Fixes #5421

PR #5426 also edits this module and its support-table entries for the differentiable path. Whichever PR
merges second should reconcile the shared histogram/threshold code and rerun the Otsu tests.

## What did you check?

Validation ran on macOS arm64 using the isolated fix worktree. The existing development environment was
reused with `UV_NO_SYNC=1`; subprocess tests used the worktree root on `PYTHONPATH`. PyTorch 2.5.1 and
2.9.1 ran in separate temporary environments. Every comparison checked the imported Kornia path.

| Check | Result |
| --- | --- |
| Full CPU float32 quick suite | 17,038 passed; 3,759 skipped; 41 xfailed; 10 non-strict xpassed; no failures |
| Filters, neighboring threshold, API surface, precision guards (CPU float32/float64) | 4,795 passed, 19 skipped |
| Final Otsu eager/strict export, CPU four floating dtypes, PyTorch 2.14.0 and 2.5.1 | 233 passed on each version |
| Final Otsu eager/strict export, MPS three floating dtypes, PyTorch 2.14.0 and 2.5.1 | 177 passed, 4 expected skips on each version |
| CPU full-graph Inductor, four floating dtypes, PyTorch 2.14.0 and 2.5.1 | 12 passed on each version |
| MPS full-graph Inductor, three floating dtypes, PyTorch 2.14.0 | 9 passed |
| Version-matched support survey, three default APIs | export and compile pass; zero graph breaks |
| `pixi run pre-commit-all`, `pixi run typecheck`, module doctest | Passed |
| `pixi run uv build` | Wheel and source distribution built |
| Installed wheel outside the source checkout | Edge, half precision, constants, and export smoke checks passed |

The final focused matrix includes the additional slow-path edge regressions added after the broader runs.
PyTorch 2.5.1 has no MPS Inductor backend; only its eager and export paths were checked on MPS.

The original source fails nine selected CPU regression cases and both MPS empty-gap cases. The issue
fixtures now keep 431 foreground pixels for the float grid and three for the rounded bfloat16 edge, and
CPU/MPS select the same occupied-bin split. Exhaustive predecessor checks cover all finite float16 and
bfloat16 bit patterns on the oldest supported compiler.

ONNX export remains unsupported: the pristine base fails export, and the changed path reaches the
exporter but has no lowering for `prims.nextafter`. This PR does not claim ONNX support. CUDA was not
available locally; the upstream CI matrix and maintainer review remain required before merging.

## How did AI help?

Codex reproduced the reports, implemented the fix and regression tests, independently reviewed the
numerical edge cases, ran the validation above, and prepared this description. The changes are submitted
as a draft for the repository owner and maintainers to review.
