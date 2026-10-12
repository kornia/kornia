`ycbcr_to_rgb` now uses the exact inverse of the `rgb_to_ycbcr` coefficients, so an
`rgb -> ycbcr -> rgb` round trip is lossless to floating-point precision instead of drifting
by ~2.7e-4. The forward transform is unchanged. (#4378)
