`distance_transform` no longer silently returns a wrong or all-zero result when its `exp(-dist/h)` kernel
underflows (#4152). The kernel's axis-aligned tap, at distance `kernel_size // 2` along a single axis,
underflowing to exactly zero -- reachable from the documented default `h=0.35` at `kernel_size >= 15` in
`float16` -- used to make the convolution silently drop it, returning a plausible-looking but wrong distance
field or, with a smaller `h`, an all-zero one. Two changes: `float16`/`bfloat16` inputs now run the cascade in
`float32` and cast the result back (the default `h=0.35` case is fixed by this alone, matching `float32`'s own
working range); and an `h`/`kernel_size`/dtype combination whose axis-aligned tap still underflows
`torch.finfo(dtype).tiny` now raises instead of guessing. This also catches a case that is not itself
numerically wrong on CPU (`h=0.01` at `kernel_size=3` in `float32` tracks the analytic reference to ~2e-4,
carried through the subnormal range) but where MPS flushes subnormals to zero while CPU does not, so the same
call used to silently disagree across backends; raising there rules out that disagreement rather than trusting
which backend a caller happens to run on. Concretely, at the documented default `h=0.35` in `float32`,
`kernel_size >= 63` now raises: CPU used to return a usable-looking answer there while MPS silently returned
something else for the same call.
