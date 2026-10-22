`kornia.color.apply_colormap` and `kornia.color.ApplyColorMap` accept empty tensors. An empty batch or an empty
channel dimension raised `RuntimeError: cannot reshape tensor of 0 elements`, and a floating image with zero height
or width raised `IndexError` from `amax`. They now return an empty tensor of the documented `(B, 3*C, H, W)` shape.
Non-empty outputs are unchanged.
