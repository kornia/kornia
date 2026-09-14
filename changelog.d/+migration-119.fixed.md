Make `yuv_to_rgb` the exact inverse of `rgb_to_yuv`, so an RGB → YUV → RGB round trip is now limited only by the
input dtype instead of losing up to `1.36e-3` (in B, at `rgb = (1, 1, 0)`) at every precision, `float64` included
(#4044). The inverse kernel was a separately rounded copy of the published BT.470-5 M/PAL inverse relations rather
than the inverse of the rounded forward kernel kornia actually ships, and one of its literals — `2.029`, where the
inverse of kornia's forward kernel is `2.03199968` — carried most of the error. `yuv_to_rgb` output therefore moves
by up to `1.23e-4` in R, `9.13e-4` in G and `1.60e-3` in B over the documented YUV domain, and `yuv420_to_rgb` and
`yuv422_to_rgb` move with it. The forward direction (`rgb_to_yuv`, `rgb_to_yuv420`, `rgb_to_yuv422`) is unchanged.
Agreement of the inverse with the standard's own relations improves overall, from `1.54e-3` to `5.24e-4`, though R
alone moves the other way (`1.54e-4` to `2.77e-4`). The `.. warning::` blocks that documented the defect on the six
affected functions and classes are gone.
