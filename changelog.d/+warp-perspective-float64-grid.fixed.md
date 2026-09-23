`warp_perspective` and `warp_perspective3d` now build their sampling grid in the input dtype, so a
`float64` image or volume is warped to `float64` roundoff instead of float32 grid precision, as
`warp_affine` already did. Half-precision grids are unchanged.
