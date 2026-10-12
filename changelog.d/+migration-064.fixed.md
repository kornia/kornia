`distance_transform` no longer returns NaN gradients when the cascade's convolution is exactly zero, including
sparse masks and all-zero inputs; the existing forward output is unchanged. (#4232)
