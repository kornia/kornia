## What does this pull request do?

Fix CUDA-only test assumptions: compare meshgrid and SOLD2 results with a one-ULP bound off CPU, keep CutMix's expected lambda on the test device with a minimal float64 rounding tolerance, and generate the orientation regression input on CPU before moving it to the requested device. `keypoints_to_grid` now evaluates the documented reference expression before reordering coordinates, avoiding CUDA rounding drift. The singleton perspective/NaN behavior from the issue is already fixed on `main`.

## Related issue or discussion

Fixes #4779

## What did you check?

- Four affected test files, CPU, PyTorch 2.14.0+cu130: 467 passed, 18 skipped, 2 xfailed.
- Four affected test files, CUDA on NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.14.0+cu130: 458 passed, 27 skipped, 2 xfailed.
- SOLD2 Dynamo test with the eager backend: 2 passed. Inductor is unavailable on this Windows environment because Triton is not installed.
- `uv build --wheel`: succeeded.
- Targeted pre-commit checks and `git diff --check` passed.

## How did AI help?

AI helped trace the issue report against current `main`, make the focused test corrections, and run the CPU/CUDA checks and build.
