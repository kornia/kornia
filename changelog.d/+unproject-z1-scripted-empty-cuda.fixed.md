`unproject_points_z1` and `compute_correspond_epilines`, compiled with `torch.jit.script` and run on CUDA, returned
an empty `int32` tensor for an empty batch (no points) once the scripted function had run on non-empty inputs:
TorchScript's CUDA fuser miscompiles a concatenation of a fused product when every input is empty. Both now build
their result without that concatenation (`unproject_points_z1` pads the points with a 1 and multiplies by the depth;
`compute_correspond_epilines` scales `F x` directly). Eager results are bitwise unchanged, and `unproject_points_z1`
also accepts a depth whose leading dimensions broadcast against the points, where it used to raise from `torch.cat`.
