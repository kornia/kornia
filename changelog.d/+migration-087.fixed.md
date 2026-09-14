`resize` and `rescale` return an empty image for a zero-sized output instead of raising
`ZeroDivisionError` out of the scale-factor computation (#4115). `resize(x, (0, 4))`, `resize(x, 0)`,
`Resize((0, 4))`, and any `rescale` factor that takes a side to zero (`rescale(x, 0.0)`) previously died
in `h / size[0]`; they now return an empty tensor of the requested shape, which is the answer
`warp_affine`, `warp_perspective` and `center_crop` already gave for the same request. The empty result
stays connected to the autograd graph, the way `_empty_warp_output_2d` builds it for the warping ops, so
a batch element that degenerates to a zero-sized output still contributes a zero gradient rather than
detaching.

A degenerate *input* is a different question and is now rejected rather than divided by: resizing a
`(1, 3, 0, 4)` image to a positive size, or with an `int` size at all -- an `int` names one side and takes
the other from an aspect ratio a degenerate input does not have -- raises
`ValueError("Input image size must be positive. Got height=0, width=4.")`, the message the warping ops
already use, instead of the same bare `ZeroDivisionError`. Empty in and empty out is well defined and
still comes back empty. (#4140)
