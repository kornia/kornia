`spatial_expectation2d`, and so `spatial_soft_argmax2d`, now builds its normalised grid in the input dtype. float64
coordinates no longer carry float32 rounding error. float16 and bfloat16 still build it in float32. (#5019)
