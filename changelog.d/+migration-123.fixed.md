Define singleton pixel axes at normalized center, reject non-positive normalization sizes, preserve empty warp
destinations, and deprecate the now-unused `eps` normalization parameters (#4006). The four
`{de,}normalize_pixel_coordinates{,3d}` helpers and `normal_transform_pixel{,3d}` now derive their scale from the
size directly instead of rounding the size into the coordinate dtype first, so results at non-degenerate sizes can
move toward the exact value: by up to one ulp in `float32`/`float64`, but materially more in `float16`/`bfloat16`,
where the old code could not represent the size at all (`denormalize_pixel_coordinates` at `bfloat16` and size 3000
returns 1496 where it returned 1504).

A singleton axis previously divided by zero (or by the `eps` substituted for it), so the change is not confined to
those helpers: it reaches every operation that normalizes a coordinate frame with a size-1 dimension.
`create_meshgrid(1, 4, normalized_coordinates=True)` returned `nan` in the singleton component and now returns `0`;
`warp_perspective`, `crop_by_transform_mat`, `homography_warp` and `warp_image_tps` onto a 1-pixel-high destination
returned all-`nan` and now return the finite row the 2-pixel control produces; `spatial_soft_argmax2d` and
`spatial_expectation2d` on a 1-pixel-high input returned `nan` and now return `0`; and `conv_soft_argmax2d`,
`conv_soft_argmax3d` and their `ConvSoftArgmax*` module forms shift by one normalized unit when a kernel or input
axis is `1` (`conv_soft_argmax2d` with a `(3, 1)` kernel returned `[-1.6667, -1.0, ...]` and now returns
`[-1.0, -0.3333, ...]`).
