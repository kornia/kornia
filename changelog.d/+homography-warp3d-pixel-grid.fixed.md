`homography_warp3d(..., normalized_coordinates=False)` now warps by a destination→source **voxel** homography: the
warped grid of source voxel coordinates is normalized under the call's `align_corners` before `grid_sample`.
Previously the voxel grid reached `grid_sample` unnormalized, so almost every sample fell outside the volume and an
identity homography returned a nearly empty volume instead of the input. An identity now returns the input and a
whole-voxel translation moves the volume by whole voxels under both `align_corners` settings, and at
`align_corners=True` the result matches `warp_perspective3d` with the inverse homography. The voxel grid is warped and
normalized in at least `float32`, also under a half default dtype and in a caller's autocast region, scripted or not,
so `float16` and `bfloat16` do not lose half or whole voxels to coordinate rounding. Callers that pre-multiplied the
homography by `normal_transform_pixel3d` to work around this must drop that factor. The default
`normalized_coordinates=True` is unchanged. Fixes #6187.
