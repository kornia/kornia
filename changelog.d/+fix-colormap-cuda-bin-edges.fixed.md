`apply_colormap` and `ApplyColorMap` now preserve eager bin selection under `torch.compile` on CUDA for floating-point images in [0, 255], including exact bin edges and non-finite values.
