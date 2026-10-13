`DenseSIFTDescriptor` output is smaller at the default `padding=1`. It used to return an `H x W` grid, one descriptor per
pixel, built from cells one pixel apart. It now returns `(H - 9) x (W - 9)`, with each output centred at input position
`j + 4.5` along each axis, and an input under 10 pixels per side raises. In general the height is
`(H + e + 2 * padding - (num_spatial_bins - 1) * spatial_bin_size - 1) // stride + 1`, where `e` is 1 for an even
`spatial_bin_size` and 0 otherwise; pass `padding=6` for `H + 1` at the other defaults.
