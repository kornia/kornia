`tilt_projection` preserves its documented leading batch dimensions, so `distort_points`,
`undistort_points`, and `undistort_image` no longer fail on multi-axis batches when tilt distortion
is applied. (#4345)
