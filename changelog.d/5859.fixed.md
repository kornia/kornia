`OnnxLightGlue(device=None)` no longer raises torch's `TypeError` from `torch.device(None)`. `None` is now
treated as `"cpu"`, as in the other kornia feature constructors, and the `device` docstring says so.
