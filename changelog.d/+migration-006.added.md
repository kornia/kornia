Documented depth and stereo conventions (the two meanings of depth, the `(B, 3, H, W)` versus
`(B, H, W, 3)` layouts, the opposite source/destination naming of `warp_frame_depth` and
`DepthWarper`, the rectified stereo `Q` matrix) and added executable pins for `kornia.geometry.depth`
and `StereoCamera`, including dtype promotion, extra-axis broadcasting, subpixel border sampling,
and singular stereo reprojection. Remaining defects are tracked in dedicated issues. (#4317)
