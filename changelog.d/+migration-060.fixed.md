`bbox_to_mask` built its pixel-position grid in the box dtype. With `float16` boxes (and
`bfloat16` boxes) on images wider or taller than 2048 px (256 px for `bfloat16`) consecutive
pixel positions collapsed, so rows and columns near the collapse were mismasked; the grid is
now built in `float32` and `float16`/`bfloat16` results are byte-identical to `float32`/`float64`.
`RandomErasing` and `RandomCutMixV2` build their masks through it. (#4336)
