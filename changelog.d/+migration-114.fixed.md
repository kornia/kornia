Drop the constant homogeneous row from `laf_to_boundary_points`. The function appended a `[0, 0, 1]` row to every
LAF so that it could divide the result by a homogeneous coordinate that is always exactly `1`; it now multiplies
by the `(2, 3)` LAF directly, and moves its 3-column basis to the LAF's device and dtype before broadcasting it
instead of after (the previous order materialized, and on an accelerator transferred, one copy per LAF -- ~12 MiB
per call at `N = 20000` for a tensor with `n_pts` distinct rows). The win is not the transfer: torch's CPU
batched gemm falls off a fast path at the third row, so `bmm` on a `(B, 3, 3)` operand costs 23x a `(B, 2, 3)` one
for 1.5x the arithmetic. Quiet back-to-back A/B against base `8db1499a` via `benchmarks/feature/laf_ops.py`
(Apple M1, torch 2.14.0, 4 threads, B=1 N=20000): `laf_to_boundary_points` 1.07M -> 29.6M LAFs/s eager on CPU and
1.77M -> 4.27M on MPS; `laf_is_inside_image`, which calls it with `n_pts=12` on every detector forward,
9.75M -> 29.0M on CPU and 2.70M -> 8.53M on MPS. Results are bitwise identical on CPU `float16` and `bfloat16`
and shift by at most 1.42 machine epsilon elsewhere, relative to the largest coordinate of the call -- the two
gemm shapes round differently. Mean error against a `float64` reference is slightly lower than before, and
`laf_is_inside_image`'s boolean masks were bitwise identical across a 126-case device/dtype/shape sweep. (#4217)
