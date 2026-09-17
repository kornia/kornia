The blocking MPS CI gate now runs on torch 2.14.0 instead of 2.9.1, on the `macos-15` image
instead of `macos-latest`, and its recorded failure baseline drops from 200 entries to 14.
torch 2.14.0 adds MPS kernels for `grid_sample`'s 2-D and 3-D backward passes, for
`padding_mode="border"` and 5-D `mode="nearest"` sampling, and for the eigen/QR/LU/SVD family
(`eigh`, `svd`, `svdvals`, `qr`, `lu_solve`, `lstsq`, `cholesky_solve`, `matrix_exp`), so most of
what used to fail on Apple silicon now runs there natively. What is still *pinned* in the
baseline: `torch.linalg.eigvals`, which has no MPS kernel and keeps the 5-point essential-matrix
solver off the device, and the `yuv420`/`yuv422` empty-input reshape defect. Separately,
`tests/geometry/test_ransac.py` is *skipped* rather than pinned — it aborts the process on the
runners' paravirtualized GPU, and a `SIGABRT` has no exception type to record
([#4204](https://github.com/kornia/kornia/issues/4204)); the skip costs 9 tests that pass on real
Apple hardware, and `--run-mps-process-abort` runs them locally. The underlying limit is that a
**batched** `torch.linalg.svd`/`svdvals`/`lstsq` fails to build a Metal pipeline once its input
holds 8192 elements or more, which used to keep `RANSAC`'s batched minimal solvers off the
device ([#4201](https://github.com/kornia/kornia/issues/4201)); `_torch_svd_cast` and
`_torch_linalg_svdvals` now route such batches through the CPU, so the fundamental and
homography solvers reach it, while `find_essential` still raises on `torch.linalg.eigvals`,
which has no MPS kernel. A single unbatched matrix is unaffected at any size. The image stays on `macos-15`: on `macos-latest` (macOS 26) the runner's *virtual*
GPU cannot compile the Metal 4 cooperative-tensor shaders torch 2.14 emits, which fails 664
tests — a physical M1 on macOS 26 compiles them fine. kornia still supports torch 2.5.1, so the
in-tree MPS workarounds stay. (#4202)
