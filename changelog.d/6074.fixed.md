`spatial_gradient` and `sobel`, and so `harris_response`, `gftt_response` and `canny`, run faster on CPU float32 and
float64 inputs with at least 128 x 128 pixels per image and 3 x 128 x 128 values in all, whose gradient is not recorded:
the first-order Sobel derivative is computed from replicate-padded shifted slices instead of a convolution, whose
buffers were fresh memory faulted in on every call. On batches of 256 x 256 RGB images, measured on an i7-14700K under
WSL2 (where page faults cost more than on native Linux), `sobel` is 5.1x faster at batch 8 and 1.5x at batch 32 in eager
mode, and 2.6x and 1.2x with the faults removed from both paths by `GLIBC_TUNABLES`; under `torch.compile` it is 8.8x and
3.5x faster. The values agree with the convolution to rounding, with the same NaN and infinity positions.
