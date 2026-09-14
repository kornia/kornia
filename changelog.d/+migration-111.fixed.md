`extract_patches_simple` and `extract_patches_from_pyramid` return an all-zero patch and a zero LAF gradient for a
non-finite LAF frame instead of handing `grid_sample` an invalid grid. A frame holding a NaN or an infinity
anywhere -- including only in its center -- is detected and sanitized before any grid arithmetic, and finite frames
in the same batch are untouched. Such a frame previously produced a finite-looking border-sampled patch, and its
backward pass could terminate the process inside torch's CPU `grid_sampler_2d_backward` kernel with
`padding_mode="border"`; a training-time detector that emits a degenerate LAF hits exactly that path through
`LAFOrienter`, `LAFAffNetShapeEstimator` and `LAFDescriptor`.
