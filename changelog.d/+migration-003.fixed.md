`kornia.enhance.equalize`, `equalize3d`, `RandomEqualize` and `RandomEqualize3D` raise a `RuntimeError` naming
the `[0, 1]` input range for values the 256-bin lookup cannot index, instead of a raw
`index 259 is out of bounds for dimension 1 with size 256` from the gather. The check uses
`torch._assert_async`, so it adds no device sync and `torch.compile` fullgraph still works, and inputs that
equalized before (including values a hair above 1) are unchanged. The docstrings now state the range, the
256-bin histogram, and that a 3D volume of at most 255 voxels per channel comes back unchanged.
(#4431, #4432, #4489)
