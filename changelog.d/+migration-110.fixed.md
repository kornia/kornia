Keep `extract_patches_simple` and `extract_patches_from_pyramid` off a broken reduced-precision CPU kernel: for a
`float16`/`bfloat16` CPU image, the `grid_sample` kernels in torch <= 2.9 read out of bounds when a rotated patch
straddles the image border and return NaN or garbage values far outside the image's range. Both extractors now
sample reduced-precision inputs in `float32` on every device and cast the patches back. This fixes the CPU kernel
bug and avoids normalized-coordinate precision loss on large images; half-precision patches change on every
backend, with the CUDA cost described under **Breaking changes** above. Both extractors also replace their
per-image Python loop with a folded batched `grid_sample` over a `(B, N*PS, PS, 2)` grid, split along `N` when
needed to bound its workspace. Both forms compile under `fullgraph=True`, but the loop was unrolled at trace time
into one `grid_sample` per batch element, so the graph grew with the batch and was recompiled for every new batch
size: tracing `extract_patches_simple` over batches of 2, 3, 5 and 7 built four graphs of 2/3/5/7 `grid_sample`
nodes before and builds two graphs with one node each now.
