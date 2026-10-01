Normalized Pascal kernels avoid half-precision overflow, preventing zero or non-finite outputs from `blur_pool2d`, `max_blur_pool2d`, and `edge_aware_blur_pool2d` with large kernels.
