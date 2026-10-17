`spatial_gradient` and `sobel`, and so `harris_response`, `gftt_response` and `canny`, run faster on CPU float32 and
float64 inputs of at least 128 x 128 pixels whose gradient is not recorded: the first-order Sobel derivative is computed
from replicate-padded shifted slices instead of a convolution, whose buffers were fresh memory faulted in on every call.
On batches of 256 x 256 RGB images, `sobel` is 5.1x faster at batch 8 and 1.5x at batch 32 in eager mode, and 8.8x and
3.5x under `torch.compile`; `harris_response` and `gftt_response` are about 1.6x faster at batch 8 in eager mode. The
values agree with the convolution to rounding, with the same NaN and infinity positions.
