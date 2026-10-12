`Image.to` and `Image.float` return a new `Image` and leave the original unchanged, like `torch.Tensor.to`; they used
to change the image in place and return it, so code that ignores the result must now assign it (`img = img.float()`).
The drawing functions place fractional coordinates on the pixel whose centre is nearest, a coordinate exactly halfway
rounding up, in `draw_point2d`, `draw_line` and `draw_rectangle`; they used to truncate toward zero, so 2.7 drew at
pixel 2 and now draws at pixel 3. `draw_line` draws, at each step along the longer axis, the pixel nearest to the
segment, so a line is the same from `p1` to `p2` as from `p2` to `p1`; it used to round the shorter-axis coordinate up
from `p1`, which put sloped lines up to one pixel off and depended on the direction. `draw_point2d` raises
`ValueError` for a point outside the image, where a coordinate of -1 or less used to draw at the opposite edge and one
past the far edge raised `IndexError`, and for an image that is not `(C, H, W)` or `(H, W)`, where a batched image was
drawn along the wrong axes or raised a torch indexing error. `draw_point2d` and `draw_line` check the bounds after
rounding, so a coordinate is inside the image from -0.5 up to, but not including, `W - 0.5` (`H - 0.5` for y): from
`W - 0.5` up to `W` both used to draw in the last column and now raise `ValueError`, `draw_point2d` also raises
between -1 and -0.5, where it drew in the first column, and `draw_line` accepts -0.5 up to 0, where it raised.
`draw_point2d` no longer exports with `torch.export`, nor to ONNX from torch 2.9, because the out-of-image check reads
tensor values; `draw_rectangle` exports with `torch.export` from torch 2.14, which it did not before. `draw_rectangle`
no longer draws a side that lies outside the image and clips a fill to the image; it used to clamp the corners, which
drew such sides along the border (clamp the corners to `[0, W - 1]` and `[0, H - 1]` for the old look). It accepts the
two corners in either order, where reversed corners drew nothing, raises `ValueError` instead of `AssertionError` when
the image and rectangle batch sizes differ or the last dimension is not 4, and raises `ShapeError`, still a
`ValueError`, naming the expected shape for an image that is not `(B, C, H, W)`. `Image` checks its pixel format:
`GRAY` needs one channel and `RGB` and `BGR` need three (`UNKNOWN` takes any count), and `bit_depth` must be the
storage size, `8 * data.element_size()`, where `bit_depth` 9 to 15 used to pass on `uint8` data and 12 failed on
`uint16`; a mismatch raises `ValueCheckError`, a `ValueError`. Without a colour space, `Image.from_numpy` and
`Image.from_dlpack` label one channel `GRAY`, three `RGB` and any other count `UNKNOWN`; they used to label every
image `RGB`. `image_to_string` and `print_image` show each pixel with the nearest of the 240 fixed xterm colours
(codes 16 to 255, the gray ramp included); they used to pick the nearest of the six colour-cube levels in each channel
after truncating a float value to a whole level, so printed colours change. `print_image` raises `TypeError` instead
of `RuntimeError` for an input that is neither a path nor a tensor. `image_to_tensor` raises `TypeError` for any input
that is not a `numpy.ndarray`, where a list raised `AttributeError`. `make_grid` raises `ValueError` for a negative
`padding`, which used to crop the images.
