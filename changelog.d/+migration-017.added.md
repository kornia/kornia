`benchmarks/feature/laf_ops.py` microbenchmarks seven public LAF operations
(`laf_from_center_scale_ori`, `make_upright`, `ellipse_to_laf`, `laf_to_boundary_points`,
`laf_is_inside_image`, `extract_patches_simple`, `extract_patches_from_pyramid`) as kornia eager vs
`torch.compile`, and its CPU and CUDA runs are the first committed non-Apple result set, so the docs
performance page now shows CUDA numbers. The page labels its throughput column from a new optional
`units` metadata key (`LAFs/s` here, `items/s` when absent) and identifies a row by every config field
that varies within a result file rather than by `(op, batch)` alone, which a suite sweeping a second
axis at a fixed batch would otherwise collapse onto one row. `run_batch_sweep` gained `label_fn`,
`items_fn` and `units` so such a suite can use the shared sweep; the existing callers are unchanged.
(#4119)
