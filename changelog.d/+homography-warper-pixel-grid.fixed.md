`homography_warp(..., normalized_coordinates=False)` and `HomographyWarper(..., normalized_coordinates=False)`, on
both the direct and the `precompute_warp_grid` path, now warp by a destination→source **pixel** homography: the
warped grid of source pixel coordinates is normalized under the call's `align_corners` before `grid_sample`, as
`warp_perspective` does. Previously the pixel grid reached `grid_sample` unnormalized, so almost every sample fell
outside the image and an identity homography returned a nearly blank image instead of the input. The result now
matches `warp_perspective` with the inverse homography. The default `normalized_coordinates=True` is unchanged.
Fixes #6170.
