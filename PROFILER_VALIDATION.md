# Representation profiler: correctness and validation draft

This work builds on navya7821's Kornia PR #3669 (head f69c19ed75f1b8b29733882ced458e78cf31e46c), associated with issue #3626. Original implementation credit belongs to that contribution. Correctness fixes, regression tests and validation were developed with substantial OpenAI Codex assistance.

## Changes

- Implement centered linear CKA using sample Gram matrices, with input and degeneracy validation.
- Return Gram matrices for every requested layer.
- Validate hook registration atomically and clone captured features to protect against in-place operations.
- Restore per-module training state after profiling, including failures.
- Integrate package exports and lazy optional pandas support.
- Add 46 profiler regression tests and usage documentation.

## Validation performed

Validation was performed on Kornia v0.8.3 with Python 3.12, PyTorch 2.6 and an RTX 4090. All 46 profiler tests passed on CPU/CUDA. Wheel build, import smoke checks and targeted lint checks passed. Reverting the fixes triggered ten regression failures.

The broader upstream runs recorded 7,724 CPU passes and 7,713 CUDA passes, with zero assertion failures after matching the original release dependency and disabling cuDNN TF32 for the affected tests. Eleven setup errors per run remained because external image fixtures were unavailable; one SAM checkpoint-dependent test was deselected. These are incomplete suite results, not an all-green suite.

## Review status

This branch is based on main commit 1adb5b93af1f95816b4dec8ee8d11cbce2581728. Existing main package initialization and project metadata were preserved while adding profiler integration. The previous test results do not establish compatibility with current main. Current-main checks, full pre-commit checks, public API typing/documentation review and agreement on the collector/metric architecture remain necessary before an upstream-ready PR. No performance or full architecture-compliance claim is made.
