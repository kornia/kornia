`unproject_meshgrid` now returns the documented `(*, H, W, 3)` shape when `W` is 1. It squeezed the
meshgrid with a bare `.squeeze()`, which drops the width axis along with the leading batch axis whenever
`W == 1`, so `unproject_meshgrid(1, 3, K)` and `unproject_meshgrid(3, 1, K)` returned the same shape with
different values. The grid then broadcast across a phantom width in the two public functions built on it:
`depth_to_3d_v2` returned `(1, 3, 3, 3)` for a `W = 1` depth where `depth_to_3d` correctly returns
`(1, 3, 1, 3)`, and `warp_frame_depth` returned a 4-pixel-wide image for a 1-pixel-wide input. Only the
batch axis is dropped now. Output for every `W > 1` is byte-identical, and `depth_to_3d_v2` agrees with
`depth_to_3d` at `W = 1` as it already did everywhere else. (#4278)
