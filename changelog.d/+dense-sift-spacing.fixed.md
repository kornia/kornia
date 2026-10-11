`DenseSIFTDescriptor` now spaces its `num_spatial_bins` x `num_spatial_bins` cells `spatial_bin_size` pixels apart, as
`SIFTDescriptor` and VLFeat's dense SIFT do. They were one pixel apart, so at the defaults one output depended on a
9 x 9 window instead of the 16 x 16 pixels of 4 cells of 4-pixel bins, and an edge anywhere in that window lit several
cell columns at once. Descriptor values change, and the output is smaller: the height is
`(H + e + 2 * padding - (num_spatial_bins - 1) * spatial_bin_size - 1) // stride + 1` (`e` is 1 for an even
`spatial_bin_size`, else 0), which is `H - 9` at the defaults (it was `H`). Pass `padding=6` for `H + 1`. The
docstring now states this size rule, the input position each output describes, and the default `padding` of 1 (it
said 0). At the defaults an image under 10 pixels per side now raises, where it used to give an output.
