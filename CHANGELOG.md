# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

****
## :rocket: [0.6.5] - 2022-05-16
### :new:  New Features
- Create `kornia.io` and implement `load_image` with rust (#1701)
- Implement `diamond_square` and plasma augmentations: `RandomPlasmaBrightness`, `RandomPlasmaContrast`, `RandomPlasmaShadow` (#1700)
- Added `RandomRGBShift` augmentations (#1694)
- Added STE gradient estimator (#1666)
- More epipolar geometry metrics (+linalg utility) (#1674)
- Add Lovasz-Hinge/Softmax losses (#1682)
- Add `adjust_sigmoid` and `adjust_log` initial implementation (#1685)
- Added distribution mapper (#1667)

### :lady_beetle: Bug fixes
- Fixes filter2d's output shape shrink when padding='same' (#1661)
- fix: added eps in geometry/rotmat_to_quaternion (#1665)
- [fix] receive num_features as an arg to KeyNetDetector constructor (#1686

### :zap:  Improvements
- Add reduction option to `MS_SSIMLoss` (#1655)
- Making epipolar metrics work with volumetric tensors (#1656)
- Add get_safe_device util (#1662)
- Added antialiasing option to Resize augmentation (#1687)
- Use nearest neighbour interpolation for masks (#1630)
- grayscale to rgb for `torch.uint8` (#1705)
- Add `KORNIA_CHECK_SAME_DEVICES` (#1775)

## :rocket: [0.6.4] - 2022-03-19
### :new:  New Features
- Adds MS-SSIMLoss reconstruction loss function (#1551)
- Added HyNet descriptor (#1573)
- Add KeyNet detector (#1574)
- Add RandomPlanckianJitter in color augmentations (#1607)
- Add Jina AI QAbot to Kornia documentation (#1628)
- Add `draw_convex_polygon` (#1636)

### :lady_beetle:  Bug fixes
- RandomCrop fix and improvement (#1571)
- Fix draw_line produce wrong output for coordinates larger than uint8
- Fix mask bug for loftr (#1580)
- Fix gradient bug for distance_transform (#1584)
- Fix translation sampling in AffineGenerator3D (#1581)
- Fix AugmentationSequential bbox keypoints transformation fix (#1570)
- Fix CombineTensorPatches (#1558)
- Fix overblur in AA (#1612)

### :exclamation: Changes
- Deprecated `return_transform`, enabled 3D augmentations in AugmentionSequential (#1590)

### :zap:  Improvements
- Making compute_correspond_epilines work with fundamental and point of volumetric tensor (#1585)
- Update batch shape when augmentations change size of image (#1609)
- Remap accepts arbitrary grid size (#1617)
- Rename variables named 'input' to 'sample' (in tests). (#1614)
- Remove half log2 in extract_patches (#1616)
- Add orientation-preserving option for AffNet and make it default (#1620)
- Add option for sampling_method in 2d perspective transform generation (#1591) (#1592)
- Fix adjust brightness (#1586)
- Added default params for laf construction from xy and new tensor shape check (#1633)
- Make nms2d jittable (#1637)
- Add fn to automatically compute padding (#1634)
- Add pillow_like option for ColorJitter to match torchvision. (#1611)

## :rocket: [0.6.3] - 2022-01-30
### :new:  New Features
- Update CI to pytorch 1.10.1 (#1518)
- Added Hanning kernel, prepare for KCF tracking (#1519)
- Add distance transform implementation (#1490)
- Add Resize augmentation module (#1545)

### :lady_beetle:  Bug fixes
- Precompute padding parameters when RandomCrop aug in container (#1494)
- Padding error with RandomCrop #1520
- Fix correct shape after cropping when forwarding parameters (#1533)
- Fixed #1534 nested augmentation sequential bug (#1536)
- Fixes to device in augmentations (#1546)
- Bugfix for larger MotionBlur kernel size ranges (#1543)
- Fix RandomErasing applied to mask keys (#1541)

### :exclamation: Changes
- Restructure augmentation package (#1515)

### :zap:  Improvements
- Add missing keepdims with fixed type (#1488)
- Allow to pass a second K to distort and undistort points (#1506)
- Augmentation Sequential with a list of bboxes as a batch (#1497)
- Adde Devcontainer for development (#1515)
- Improve the histogram_matching function (#1532)

## :rocket: [0.6.2] - 2021-12-03
### :new:  New Features
- Add face detection API (#1469)
- Add `ObjectDetectorTrainer` (#1414)
- Add container operation weights and `OneOf` documentation (#1443)
- Add oriented constraint check to Homography RANSAC (#1453)
- Add background color selection in `warp_perspective` (#1452)
- Add `draw_line` image utility (#1456)
- Add Bounding Boxes API (#1304)
- Add histogram_matching functionality (#1395)

### :lady_beetle:  Bug fixes
- fix catch type for torch.svd error (#1431)
- Fix for nested AugmentationSequential containers (#1467)
- Use common bbox format xywh (#1472)
- Fix motion blur kernel size bug for larger random generator ranges (#1540)

### :exclamation: Changes
- Add padding_mode for RandomElasticTransform augmentation (#1439)
- Expose inliers sum to HomographyTracker (#1463)

### :zap:  Improvements
- Switch to one-way error RANSAC for speed-up (#1454)
- Few improvements on homography tracking (#1434)
- Enable all bandit tests, add separate hook for tests (#1437)
- Merge homography_warp to warp_perspective (#1438)
- Random generator refactor (#1459)


## :rocket: [0.6.1] - 2021-10-22
### :lady_beetle:  Bug fixes
- Fixes PyPI tarball missing required files #1421
- hotfix: remove mutable object in constructor #1423


## :rocket: [0.6.0] - 2021-10-22

### :new:  New Features
- Add Training API (#1307)
- Added combine patches (#1309)
- Add semantic segmentation trainer (#1323)
- Add vanilla LO-RANSAC (#1335)
- Add Lambda function module (#1346)
- Add support for YUV420 and YUV422 to complement current YUV444 (#1360)
- Add raw to rgb color conversion (#1380)
- Implement separable_filter2d (#1385)
- Add MobileViT to contrib (#1388)
- Add solve_pnp_dlt (#1349)
- Add function image_list_to_tensor to utils (#1393)
- Add undistort_image function (#1303)
- Create kormia.metrics submodule (#1325)
- Add Image Stitching API (#1358)
- Add Homography Tracker API (#1389)

### :exclamation: Changes
- Refactor library namespaces [pre-release][0.6-rc1] (#1412)
- deprecate 1.6/1.7 and add 1.9.1 (#1399)

### :zap:  Improvements
- Improve bbox_to_mask (#1351)
- Refactor unfold->conv for morphology backbone (#1107)
- Improve focal loss for numerical stability (#1362)
- Add more border_type options for filter2D (#1375)
- Replace deprecated torch.qr (#1376)
- Add special case hardcoded implementtion for local features speed up (#1387)
- Enable non/batched connected components (#1193)
- Remove warnings during testing (#1401)

### :lady_beetle:  Bug fixes
- Fix binary focal loss (#1313)
- Fix kornia.geometry.subpix.spatial_soft_argmax imports (#1318)
- Fixed a simple typo in __init__.py (#1319)
- Fix path to dev requirements file in a setup_dev_env.sh (#1324)
- Fix bug in create_meshgrid3d along depth (#1330)
- Fix anisotropic scale error (#1340)
- Fix rgb_to_hsv for onnx (#1329)
- Fixed useless return in ransac.py (#1352)
- Fixed classificationhead typo and leave out some of the guesswork (#1354)
- Fix clahe differentiability and tests (#1356)
- Fixes singular matrix inverse/solve for RANSAC and ConvQuad3d (#1408)
- Change intermediate datatype to fix imgwarp (#1413)

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
