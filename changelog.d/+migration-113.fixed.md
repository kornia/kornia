Replace the batched `torch.inverse` in `ellipse_to_laf` with the closed-form inverse of its lower-triangular
2x2 matrix. Results agree with the previous implementation to ~1.6e-7 relative in `float32` (machine epsilon in
`float64`). Reproducible numbers via `benchmarks/feature/ellipse_to_laf.py` (float32, N=1e3..1e5, torch 2.9.1,
Linux/WSL2, base `2009933e`): throughput improves ~2.4-5.3x eager / ~3-6.5x compiled on CPU (i7-14700K) and
~1.4-1.7x eager / ~4.2-5.5x compiled on CUDA (RTX 4090). The largest win is MPS, where the batched inverse hit a
pathological `linalg` path, emitted a deprecated-resize `UserWarning` on every call, and failed to `torch.compile`
(~1500x once measured ad hoc at N=20000 on Apple Silicon; rerun the benchmark there for a citable number). The
function now compiles with `fullgraph=True` on every backend and accepts `float16`/`bfloat16` on CPU, where the
raw `.inverse()` call raised for low-precision dtypes (`_torch_inverse_cast` would have lifted that restriction
too; the closed form is for speed). The off-diagonal divides `a21 = b / (a11 + a22)` by the product of the
diagonal square roots, which can neither overflow nor round to zero: every ordering of the reciprocal product
overflows to `inf` (or flushes a representable value to `0`) on a sufficiently lopsided or subnormal diagonal,
and when that product is subnormal the division loses precision but never corrupts a representable result to a
zero, `inf`, or `nan`.
