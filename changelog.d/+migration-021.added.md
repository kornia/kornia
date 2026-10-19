Dynamo ONNX export (`torch.onnx.export(..., dynamo=True)`) now covers a much larger part of the
library: about 140 additional operator cases in the export survey capture, pass the ONNX checker
and match eager execution under onnxruntime. Operators that failed graph capture or had no ONNX
lowering gained an export-time path that returns the same values eagerly: matrix inverses up to
4x4 use a closed-form adjugate instead of `aten::linalg_inv` (`warp_perspective3d`,
`warp_affine3d`, `rotate3d`, `affine3d`, `get_projective_transform`, `invert_affine_transform`,
`denormalize_homography`, `symmetric_transfer_error`, `safe_inverse_with_mask`, `DepthWarper`,
`PinholeCamera`, `StereoCamera`, `NamedPose`, `ImageRegistrator`, `ParametrizedLine.intersect`
and the `Se2`/`Se3`/`So2` groups); `MedianBlur`/`median_blur` select the median through `sort`;
`HausdorffERLoss`/`HausdorffERLoss3D` pool through `amax` (which also makes the 3D loss run on MPS); `distance_transform` avoids `hypot`;
`match_nn`/`match_mnn`/`DescriptorMatcher` avoid `cdist`; the mutual-information losses, the
Bessel kernels, `solve_quadratic`, `matrix_cofactor_tensor`, `decompose_essential_matrix_no_svd`,
`distort_points`/`undistort_points`/`undistort_image`, `Boxes`/`Boxes3D` (`from_tensor`,
`to_mask`), `bbox_to_mask3d`, `mean_iou_bbox`, `conv_soft_argmax2d` with a tensor temperature,
`JPEGCodecDifferentiable`, `ZCAWhitening` (`transform`/`inverse_transform`), `BoxFiltering`,
`MotionBlur3D`, `center_crop`/`crop_and_resize`/`crop_by_transform_mat3d` and the
`OriNet`/`LAFOrienter`/`LAFAffNetShapeEstimator`/`HardNet`/`HardNet8`/`SIFTFeature`/
`LAFDescriptor`/`DeDoDe` feature stack are branch-free or guard their data-dependent checks with
`kornia.core.utils.is_exporting()`. Among augmentations, `LongestMaxSize`, `SmallestMaxSize`,
`RandomCrop`, `RandomResizedCrop`, `RandomAffine3D`, `RandomRotation3D`, `RandomMotionBlur3D`,
`RandomJPEG`, `RandomMedianBlur`, `RandomPlanckianJitter` and `AugmentationSequential` with
crops or boxes now export, and `LightGlue` exports once `depth_confidence`/`width_confidence`
pruning is disabled. Behavior notes: `Se2`/`Se3`/`So2`/`ParametrizedLine` built from a tensor
that already carries autograd history (`Se2.exp(v)`, `ParametrizedLine.through(p0, p1)`) keep
that history: the tensor is registered as a buffer instead of being re-rooted as an
`nn.Parameter`, so `.to()`, `state_dict()` and `load_state_dict()` still reach it under the same
key but it no longer appears in `parameters()`; `kornia.core.utils.is_exporting()` falls back to
`is_compiling()` on torch < 2.6, which has no export flag (inside a Dynamo trace newer torch
folds its flag to `True` for `torch.compile` as well, so the export-safe paths are what a
compiled graph contains on every supported version); under export `crop_by_indices` with a
`size` resamples the crop and without one keeps the eager path, `distort_points`/
`undistort_points` always apply the tilt term (a no-op when the coefficients are zero) and
`normalize`/`denormalize` accept a 1-D `(C,)` mean/std; `matrix_cofactor_tensor` returns exact
cofactors for singular matrices instead of raising when every matrix in the batch is singular.
(#4182)
