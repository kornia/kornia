# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## :rocket: [0.5.11] - 2021-08-30
### :new:  New Features
- Add Vision Transformer (ViT) ([#1296](https://github.com/kornia/kornia/pull/1296))
- Add ImageRegistrator API ([#1253](https://github.com/kornia/kornia/pull/1253))
- Add LoFTR inference ([#1218](https://github.com/kornia/kornia/pull/1218))
- Added differentiable Hausdorff Distance (HD) loss ([#1254](https://github.com/kornia/kornia/pull/1254))
- Add PadTo to kornia.augmentation ([#1286](https://github.com/kornia/kornia/pull/1286))

### :zap:  Code refactor
- Return all learned modules by default in eval() mode ([#1266](https://github.com/kornia/kornia/pull/1266))
- Enable ImageSequential and VideoSequential to AugmentationSequential (#1231)
- Specify that angles are in radians ([#1287](https://github.com/kornia/kornia/pull/1287))
- Removed deprecated codes for v6.0 ([#1281](https://github.com/kornia/kornia/pull/1281))

### :lady_beetle:  Bug fixes
- Fix save_pointcloud_ply fn counting point with inf coordinates ([#1263](https://github.com/kornia/kornia/pull/1263))
- Fixes torch version parse and add temporal packaging dependency ([#1284](https://github.com/kornia/kornia/pull/1284))
- Fix issue of image_histogram2d ([#1295](https://github.com/kornia/kornia/pull/1295))


## [0.5.10] - 2021-08-30

### Added
- Added Basic pool request for DeFMO. ([#1135](https://github.com/kornia/kornia/pull/1135))
- Added homography error metrics, and improved find_homography_iter ([#1222](https://github.com/kornia/kornia/pull/1222))

### Fixed
- Fixed wrong param name ([#1197](https://github.com/kornia/kornia/pull/1197))
- Fixed NotImplementedError for the rtvec ([#1215)](https://github.com/kornia/kornia/pull/1215))
- Fixes warnings and add compatibility stub in torch solve ([#1235](https://github.com/kornia/kornia/pull/1235))

### Changed
- Ensure CenterCrop indices are integers ([#1208](https://github.com/kornia/kornia/pull/1208))
- Added tests, fixed docstrings and made some other changes ([#1211](https://github.com/kornia/kornia/pull/1211))
- Upgrade to modern Python syntax ([#1213](https://github.com/kornia/kornia/pull/1213))
- Code health improvements [#1199, #1200, #1198, #1202, #1203, #1205, #1208, #1210, #1214, #1220]
- Enable pyupgrade as pre-commit ([#1221](https://github.com/kornia/kornia/pull/1221))
- Add bandit tool in the pre-commit ([#1228](https://github.com/kornia/kornia/pull/1228))


## [0.5.8] - 2021-08-06

### Added
- Add the connected components labeling algorithm ([#1184](https://github.com/kornia/kornia/pull/1184))

### Fixed
- Partial fix for horizontal and vertical flips ([#1166](https://github.com/kornia/kornia/pull/1166))
- Fix even kernel and add test ([#1183](https://github.com/kornia/kornia/pull/1183))
- Fix wrong source points for RandomThinPlateSpline ([#1187](https://github.com/kornia/kornia/pull/1187))
- Fix RandomElasticTransform ignores same_on_batch ([#1189](https://github.com/kornia/kornia/pull/1189))
- Fixed bugs in patchsequential. Remove fill_diagonal operation for better ONNX support ([#1178](https://github.com/kornia/kornia/pull/1178))

### Changed
- Differentiable image histogram using kernel density estimation ([#1172](https://github.com/kornia/kornia/pull/1172))


## [0.5.7] - 2021-07-27

### Added
- Grayscale to RGB image conversion. ([#1162](https://github.com/kornia/kornia/pull/1162))
- Add keepdim param to tensor_to_image function.  ([#1168](https://github.com/kornia/kornia/pull/1168))

### Fixed
- Fix checks on wrong tensor shape condition in depth.py ([#1164](https://github.com/kornia/kornia/pull/1164))


## [0.5.6] - 2021-07-12

### Added
- Added mix augmentations in containers ([#1139](https://github.com/kornia/kornia/pull/1139))

### Fixed
- Fixed non-4-dim input error for sequential ([#1146](https://github.com/kornia/kornia/pull/1146))

### Changed
- Moving bbox-related functionality to bbox module ([#1103](https://github.com/kornia/kornia/pull/1103))
- Optimized version of hls_to_rgb and rgb_to_hls ([#1154](https://github.com/kornia/kornia/pull/1154))

### Removed
- Remove numpy dependency ([#1136](https://github.com/kornia/kornia/pull/1136))


## [0.5.5] - 2021-06-26

### Added
- Added Stereo camera class ([#1102](https://github.com/kornia/kornia/pull/1102))
- Added auto-generated images in docs ([#1105](https://github.com/kornia/kornia/pull/1105)) ([#1108](https://github.com/kornia/kornia/pull/1108)) ([#1127](https://github.com/kornia/kornia/pull/1127)) ([#1128](https://github.com/kornia/kornia/pull/1128)) ([#1129](https://github.com/kornia/kornia/pull/1129)) ([#1131](https://github.com/kornia/kornia/pull/1131))
- Added chinese version README ([#1112](https://github.com/kornia/kornia/pull/1112))
- Added random_apply to augmentaton containers ([#1125](https://github.com/kornia/kornia/pull/1125))

### Changed
- Change GaussianBlur to RandomGaussianBlur ([#1118](https://github.com/kornia/kornia/pull/1118))
- Update ci with pytorch 1.9.0 ([#1120](https://github.com/kornia/kornia/pull/1120))
- Changed option for mean and std to be tuples in normalization ([#987](https://github.com/kornia/kornia/pull/987))
- Adopt torch.testing.assert_close ([#1031](https://github.com/kornia/kornia/pull/1031))

### Removed
- Remove numpy import ([#1116](https://github.com/kornia/kornia/pull/1116))


## [0.5.4] - 2021-06-11

### Added
- Add Canny edge detection ([#1020](https://github.com/kornia/kornia/pull/1020))
- Added Batched forward function ([#1058](https://github.com/kornia/kornia/pull/1058))
- Added denormalize homography function [(#1061](https://github.com/kornia/kornia/pull/1061))
- Added more augmentations containers ([#1014](https://github.com/kornia/kornia/pull/1014))
- Added calibration module and Undistort 2D points function ([#1026](https://github.com/kornia/kornia/pull/1026))
- Added patch augmentation container ([#1095](https://github.com/kornia/kornia/pull/1095))

### Fixed
- Remove lena ([#1059](https://github.com/kornia/kornia/pull/1059)) :)

### Changed
- Resize regardless of number of dims, considering the last two dims as image ([#1047](https://github.com/kornia/kornia/pull/1047))
- Raise error if converting to unit8 image to gray with float weights ([#1057](https://github.com/kornia/kornia/pull/1057))
- Filter 2D->2d, 3D->3d ([#1069](https://github.com/kornia/kornia/pull/1069))
- Removed augmentation functional module. ([#1067](https://github.com/kornia/kornia/pull/1067))
- Make Morphology compatible with both OpenCV and Scipy ([#1084](https://github.com/kornia/kornia/pull/1084))


## [0.5.3] - 2021-05-29

### Added
- Added inverse for augmentations ([#1013](https://github.com/kornia/kornia/pull/1013))
- Add advanced augmentations: RandomFisheye, RandomElasticTransform, RandomThinPlateSpline, RandomBloxBlur ([#1015](https://github.com/kornia/kornia/pull/1015)

### Fixed
- Correct Sobel test_noncontiguous. Nothing was tested before. ([#1018](https://github.com/kornia/kornia/pull/1018))
- Fixing #795: find_homography_dlt_iterated sometimes fails ([#1022](https://github.com/kornia/kornia/pull/1022))

### Changed
- Refactorization of the morphology package ([#1034](https://github.com/kornia/kornia/pull/1034))
- Optimised clipping in clahe and some other minor optimisation ([#1035](https://github.com/kornia/kornia/pull/1035))


## [0.5.2] - 2021-05-14

## Added
- Added unsharp mask filtering ([#1004](https://github.com/kornia/kornia/pull/1004))

### Fixed
- Fixed angle axis to quaternion order bug ([#926](https://github.com/kornia/kornia/pull/926))
- Fixed type error for lab_to_rgb conversion when using coremltools. ([#1002](https://github.com/kornia/kornia/pull/1002))

### Changed
- Mask with unbatched motion from essential choose solution ([#998](https://github.com/kornia/kornia/pull/998))


## [0.5.1] - 2021-04-30

### Added
- Added dtype for create_mesh ([#919](https://github.com/kornia/kornia/pull/919))
- Added Hardnet8 ([#955](https://github.com/kornia/kornia/pull/955))
- Added normalize boolean for remap ([#921](https://github.com/kornia/kornia/pull/921))
- Added custom weights option for rgb2gray ([#944](https://github.com/kornia/kornia/pull/944))
- Added fp16 support ([#963](https://github.com/kornia/kornia/pull/963))
- Added ImageToTensor module and resize for non-batched images ([#978](https://github.com/kornia/kornia/pull/978))
- Add more augmentations ([#960](https://github.com/kornia/kornia/pull/960))
- Anti alias resize ([#989](https://github.com/kornia/kornia/pull/989))

## Changed
- Improve kornia porphology ([#965](https://github.com/kornia/kornia/pull/965))
- Improve cuda ci workflow speed ([#975](https://github.com/kornia/kornia/pull/975))
- Refactor augmentation module ([#948](https://github.com/kornia/kornia/pull/948))
- Implement fast version of crop function in augmentations ([#967](https://github.com/kornia/kornia/pull/967))
- Implement missing jit ops in kornia.geometry.transform ([#981](https://github.com/kornia/kornia/pull/981))

### Fixed
- Fixed RandomAffine translation range check ([#917](https://github.com/kornia/kornia/pull/917)
- Fixed the issue of NaN gradients by adding epsilon in focal loss ([#924](https://github.com/kornia/kornia/pull/924))
- Allow crop size greater than input size. ([#957](https://github.com/kornia/kornia/pull/957))
- Fixed RandomCrop bug ([#951](https://github.com/kornia/kornia/pull/951))

### Removed
-  Deprecate some augmentation functionals ([#943](https://github.com/kornia/kornia/pull/943))


## [0.4.1] - 2020-10-20
### Added
- Update docs for `get_affine_matrix2d` and `get_affine_matrix3d` ([#618](https://github.com/kornia/kornia/pull/618))
- Added docs for `solarize`, `posterize`, `sharpness`, `equalize` ([#623](https://github.com/kornia/kornia/pull/623))
- Added tensor device conversion for solarize params ([#624](https://github.com/kornia/kornia/pull/624))
- Added rescale functional and transformation ([#631](https://github.com/kornia/kornia/pull/631))
- Added Mixup data augmentation ([#609](https://github.com/kornia/kornia/pull/609))
- Added `equalize3d` ([#639](https://github.com/kornia/kornia/pull/639))
- Added `decompose 3x4projection matrix` ([#650](https://github.com/kornia/kornia/pull/650))
- Added `normalize_min_max` functionality ([#684](https://github.com/kornia/kornia/pull/684))
- Added `random equalize3d` ([#653](https://github.com/kornia/kornia/pull/653))
- Added 3D motion blur ([#713](https://github.com/kornia/kornia/pull/713))
- Added 3D volumetric crop implementation ([#689](https://github.com/kornia/kornia/pull/689))
  - `warp_affine3d`
  - `warp_perspective3d`
  - `get_perspective_transform3d`
  - `crop_by_boxes3d`
  - `warp_grid3d`


### Changed
- Replace convolution with `unfold` in `contrib.extract_tensor_patches` ([#626](https://github.com/kornia/kornia/pull/626))
- Updates Affine scale with non-isotropic values ([#646](https://github.com/kornia/kornia/pull/646))
- Enabled param p for each augmentation ([#664](https://github.com/kornia/kornia/pull/664))
- Enabled RandomResizedCrop batch mode when same_on_batch=False ([#683](https://github.com/kornia/kornia/pull/683))
- Increase speed of transform_points ([#687](https://github.com/kornia/kornia/pull/687))
- Improves `find_homography_dlt` performance improvement and weights params made optional ([#690](https://github.com/kornia/kornia/pull/690))
- Enable variable side resizing in `kornia.resize` ([#628](https://github.com/kornia/kornia/pull/628))
- Added `Affine` transformation as `nn.Module` ([#630](https://github.com/kornia/kornia/pull/630))
- Accelerate augmentations ([#708](https://github.com/kornia/kornia/pull/708))

### Fixed
- Fixed error in normal_transform_pixel3d ([#621](https://github.com/kornia/kornia/pull/621))
- Fixed pipelining multiple augmentations return wrong transformation matrix (#645)([645](https://github.com/kornia/kornia/pull/645))
- Fixed flipping returns wrong transformation matrices ([#648](https://github.com/kornia/kornia/pull/648))
- Fixed 3d augmentations return wrong transformation matrix ([#665](https://github.com/kornia/kornia/pull/665))
-  Fix the SOSNet loading bug ([#668](https://github.com/kornia/kornia/pull/668))
- Fix/random perspective returns wrong transformation matrix ([#667](https://github.com/kornia/kornia/pull/667))
- Fixes Zca inverse transform ([#695](https://github.com/kornia/kornia/pull/695))
- Fixes Affine scale bug ([#714](https://github.com/kornia/kornia/pull/714))

## Removed
- Removed `warp_projective` ([#689](https://github.com/kornia/kornia/pull/689))
