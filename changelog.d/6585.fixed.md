`otsu_threshold` and `OtsuThreshold` under `torch.compile` on CUDA now put a pixel that lies on a bin edge in the same
bin as eager. Triton's approximate float32 division used to place it in the bin below, which changed the mask at that
pixel and could move the threshold by a whole bin.
