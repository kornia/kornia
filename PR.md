## What does this pull request do?

Fix CUDA-only assumptions in tests: allow one-ULP differences for meshgrid/SOLD2 reference comparisons, keep CutMix's expected lambda on the test device with a minimal float64 rounding tolerance, and generate the orientation regression input on CPU before moving it to the requested device. The singleton perspective/NaN behavior from the issue is already fixed on `main`.

This is test-only; please apply the `no-changelog` label.

## Related issue or discussion

Fixes #4779

## What did you check?

- Four affected test files, CPU, PyTorch 2.14.0+cu130: 467 passed, 18 skipped, 2 xfailed.
- Four affected test files, CUDA on NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.14.0+cu130: 458 passed, 27 skipped, 2 xfailed.
- `uv build --wheel`: succeeded.
- Targeted pre-commit checks and `git diff --check` passed.

## How did AI help?

AI helped trace the issue report against current `main`, make the focused test corrections, and run the CPU/CUDA checks and build.
