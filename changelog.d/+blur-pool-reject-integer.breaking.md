`blur_pool2d`, `max_blur_pool2d`, `edge_aware_blur_pool2d`, `BlurPool2D`, `MaxBlurPool2D` and `EdgeAwareBlurPool2D`
raise a `TypeCheckError` (a `TypeError`) for an integer or bool input, naming the dtype. The blur pools are network
layers and expect a floating feature map; unlike the image filters (#5669) they do not promote integer input. Before,
the binomial kernel was cast to the input's dtype, which truncated every tap to 0. On CPU, `BlurPool2D` and
`MaxBlurPool2D` returned an all-zero integer tensor without an error, and the functions raised a `RuntimeError` dtype
mismatch. On CUDA, every entry point raised a `RuntimeError` from cuDNN or the max pool. `except RuntimeError` no
longer catches the error. The check is skipped under `disable_checks()`. Floating input is unchanged.
