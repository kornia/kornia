`ssim` and `SSIM`, and `ssim_loss` and `SSIMLoss` evaluated without a recorded gradient, run faster on the CPU for two
contiguous images computed in float32 (float32, half-precision and integer images): the five Gaussian filterings now run
on channels-last copies of the images, which oneDNN convolves directly, and the local moments are combined in place.
Previously each of the ten depthwise convolutions copied its NCHW input into blocks of 8 or 16 channels and back,
buffers 2.7 to 5.3 times an RGB batch that a large batch got as fresh memory on every call. On batches of 256 x 256 RGB
images, measured on an i7-14700K under WSL2 (where page faults cost more than on native Linux), `ssim` is 2.9x faster at
batch 8 and 32, and 2.0x to 2.3x at batch 8 and 1.4x to 2.0x at batch 32 with the faults removed from both paths by
`GLIBC_TUNABLES`. The map was bitwise equal to the NCHW path's in every case measured. A call that records a gradient,
float64, channels-last and other non-contiguous images, CUDA and MPS inputs, and exported and traced graphs keep the
NCHW convolutions; under `torch.compile`, a channels-last reverse pass was slower than Inductor's own layout choice.
