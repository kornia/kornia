`kornia.contrib.histogram_matching` sums the histogram counts in `int64` and computes the quantiles in
`float32` (`float64` for a `float64` source) instead of in the input dtype. A `uint8` sum wrapped past 255,
so a 50x50 `uint8` image mapped to values above 255 and a 64x64 one to NaN everywhere; `float16` and
`bfloat16` sums stopped being exact past 2048 and 256 pixels, and about half of random image pairs at
those sizes returned NaN pixels. A floating source keeps its dtype; an integer source returns `float32`.
