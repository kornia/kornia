`otsu_threshold` / `OtsuThreshold` no longer return a threshold of 0 for a `float16` image plane of 65520
pixels or more (256 x 256 and up): the histogram counts were cast back to the image dtype, so their sum
overflowed to `inf`, and `bfloat16` counts were rounded to 8 bits. The counts are now kept in
`float32` (`float64` for a `float64` image). `normalize_homography`, and the `warp_perspective` and
`warp_affine` calls built on it, now invert `float16` and `bfloat16` matrices in `float32` and cast the
result back: the `float16` determinant of a large image's pixel-normalization matrix is subnormal, so a
3000 px `float16` normalization was 7 % off and an 11600 px one was NaN, and `bfloat16` on MPS with
torch 2.5.1 raised for want of a `cross` kernel.
