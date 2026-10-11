The kernel builders raise for an integer or bool `dtype` that cannot hold their taps. `get_box_kernel1d`,
`get_box_kernel2d`, `get_gaussian_kernel1d`, `get_gaussian_erf_kernel1d`, `get_gaussian_discrete_kernel1d`,
`get_gaussian_kernel2d`, `get_gaussian_kernel3d`, `gaussian`, `gaussian_discrete`, `gaussian_discrete_erf`,
`get_diff_kernel3d`, `get_diff_kernel3d_2nd_order` and `get_spatial_gradient_kernel3d` have fractional taps and raise
for every integer `dtype`; they used to return a kernel with its fractional taps truncated (all zeros for the box and
3-D first-order kernels) and, for uint8 Gaussians, the negative offsets wrapped. `get_laplacian_kernel1d`,
`get_laplacian_kernel2d` and `laplacian_1d` keep accepting a signed integer `dtype` wide enough for their centre tap
and raise for uint8, bool and a narrower one, where they used to wrap the centre (to 252 for a uint8 size 5).
`spatial_gradient3d` and `SpatialGradient3d` now return float32 for an integer `order=2` input, which used to return
the input's integer dtype.
