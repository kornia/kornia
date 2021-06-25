# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
