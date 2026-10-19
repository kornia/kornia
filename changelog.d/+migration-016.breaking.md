`MKDDescriptor` runs in `float16` and `bfloat16`. Its gradient-embedding and spatial-encoding stages were held in
a plain `dict`, so `.to(device, dtype)` never reached their buffers and a half-precision input failed at the
whitening matmul with `expected m1 and m2 to have the same dtype`. They are now an `nn.ModuleDict`; float32 and
float64 outputs are byte-identical, and `state_dict()` gains the stages' buffer keys (`feats.<parametrization>.*`).
A state dict saved by an earlier release still loads with `strict=True`: those buffers are derived from the
constructor arguments, and when a state dict has no `feats.*` key at all they are filled in from the module being
loaded into (one that has some and lost one is reported by `strict=True`, as for any other missing key). The
reverse is not covered: a state dict saved by this release carries the `feats.*` keys and fails `strict=True` on
an earlier kornia, which does not know them; load it with `strict=False` there.
