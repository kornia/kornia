`create_meshgrid` and `create_meshgrid3d` return different normalized coordinates at `float16`
and `bfloat16`. They used to evaluate `(xs / (size - 1) - 0.5) * 2` as three half-precision ops
in eager, rounding after each, and to materialize the half ramp before the graph-capture branch
widened it; `torch.compile` instead computed the chain in `float32` and rounded once, and folded
the narrow-then-widen round trip away so that above `2 ** p` the two paths divided different
numerators. A half-precision grid therefore changed the moment a model was wrapped in
`torch.compile` -- by up to 0.0117 in the `[-1, 1]` range at `bfloat16` width 300 and 0.0156 at
width 6000. A normalized half-precision grid is now built and normalized in `float32` and
narrowed once, so eager, traced and compiled grids agree; the eager values move onto the
correctly rounded side, which is the numeric change. `float32`, `float64` and integer grids and
the pixel ramp are bit-for-bit unchanged, since none of them widen. On CUDA at `float16` up to
one ulp of the coordinate dtype can still differ, because ATen's and triton's `float32` division
are not bit-identical; `bfloat16` absorbs it and the CPU kernels agree exactly. The Linux CPU
half-precision baselines move with it. `HomographyWarper` builds its precomputed grid in `float32`
and its on-the-fly grid in the input dtype, and the two now agree, which fixes
`test_homography_warper` at both dtypes and `test_translation[shape1]` at `float16`. At `bfloat16`
`test_translation[shape0]` and `[shape2]` join the `shape1` and `shape3` entries already recorded:
that test's `atol=1e-4` is out of reach for all four of its shapes there, and the two that passed
did so only because the old grid's larger error round-tripped through `grid_sample`'s `bfloat16`
unnormalize as an exact pixel index. Refs #4030. (#4231)
