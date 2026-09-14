Raise `NotImplementedError` instead of `RuntimeError` for an integral `src` on the empty-`dsize` paths of
`warp_affine`, `warp_perspective`, `remap`, `warp_affine3d` and `warp_perspective3d`, matching what the non-empty
path already raises from `grid_sample` (#4031). The degenerate path had its own explicit guard that fired first
with a different exception type, so the exception a caller saw for the same invalid input depended on whether
`dsize` had a zero dimension. The message is unchanged. `NotImplementedError` derives from `RuntimeError`, so
`except RuntimeError` around these functions still catches it; only code matching on the exact type sees a
difference.
