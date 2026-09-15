`kornia.color.rgb_to_yuv420` subsamples chroma with `avg_pool2d` instead of a mean over two unfolded window dimensions: same values, several times faster on CPU and MPS.
