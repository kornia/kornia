# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

****


## :rocket: [0.6.11] - 2022-03-28
### :new:  New Features

* add `DISK` local feature by @jatentaki  in https://github.com/kornia/kornia/pull/2285
* Add Joint Bilateral Filter by @gau-nernst  https://github.com/kornia/kornia/pull/2244
* Add Bilateral Filter by @gau-nernst  https://github.com/kornia/kornia/pull/2242
* Add random snow by @just1ce415  https://github.com/kornia/kornia/pull/2229


## :rocket: [0.6.10] - 2022-02-17
### :new:  New Features

* add `depth_from_disparity` function by @pri1311 in https://github.com/kornia/kornia/pull/2096
* Add Vector2 by @cjpurackal in https://github.com/kornia/kornia/pull/2134
* Add 3D-SSIM loss by @pri1311 in https://github.com/kornia/kornia/pull/2130
* [Feat] Initiate AutoAugment modules by @shijianjian in https://github.com/kornia/kornia/pull/2181
* Add Common Regression Losses by @ChristophReich1996 in https://github.com/kornia/kornia/pull/2109
* Add `integral_image` and `integral_tensor` by @AnimeshMaheshwari22 in https://github.com/kornia/kornia/pull/1779


### :lady_beetle: Bug fixes

* Fix AugmentationSequential to return list of boxes by @johnnv1 in https://github.com/kornia/kornia/pull/2114
* Fix support for (*, 3, H, W) tensors  in yuv by @ChristophReich1996 in https://github.com/kornia/kornia/pull/2108
* fix TensorWrapper serialization by @edgarriba in https://github.com/kornia/kornia/pull/2132
* Split the half precision tests workflow by @johnnv1 in https://github.com/kornia/kornia/pull/2118
* Fixed DoG accuracy, add `upscale_double` by @vicsyl in https://github.com/kornia/kornia/pull/2105
* Added Face detection Interactive demo by @jeffin07 in https://github.com/kornia/kornia/pull/2142
* Bump pytest from 7.2.0 to 7.2.1 by @dependabot in https://github.com/kornia/kornia/pull/2148
* add SSIM3D and `depth_from_disparity` to docs by @pri1311 in https://github.com/kornia/kornia/pull/2150
* Explicitly cast output to input type to avoid type mismatch errors by @JanSellner in https://github.com/kornia/kornia/pull/1842
* Fix params computation for `LongestMaxSize` and `SmallestMaxSize` by @johnnv1 in https://github.com/kornia/kornia/pull/2131
* torch_version_geq -> torch_version_ge according to todo by @ducha-aiki in https://github.com/kornia/kornia/pull/2157
* fix doc build - `sphinx-autodoc-typehints==1.21.3` by @johnnv1 in https://github.com/kornia/kornia/pull/2159
* ScaleSpaceDetector -> Fast ScaleSpaceDetector by @ducha-aiki in https://github.com/kornia/kornia/pull/2154
* Improve losses tests, add `TestSSIM3d`, and `BaseTester.gradcheck` by @johnnv1 in https://github.com/kornia/kornia/pull/2152
* modify comments of rgb and lab conversion by @gravitychen in https://github.com/kornia/kornia/pull/2153
* add __repr__ and __getitem__ to vector by @cjpurackal in https://github.com/kornia/kornia/pull/2163
* Fix adalam-config by @ducha-aiki in https://github.com/kornia/kornia/pull/2170
* Fix docs  of `boxes`, `MultiResolutionDetector`. `apply colormap`, `AugmentationSequential` by @johnnv1 in https://github.com/kornia/kornia/pull/2167
* add exception test for se2 + small bug fix by @cjpurackal in https://github.com/kornia/kornia/pull/2160
* Fix MobileViT by @chinhsuanwu in https://github.com/kornia/kornia/pull/2172
* Fix output types of augmentations on autocast regions by @johnnv1 in https://github.com/kornia/kornia/pull/2168
* Fix planckian jitter for cuda by @johnnv1 in https://github.com/kornia/kornia/pull/2177
* Fix: resample method None default missing for inverse masks by @miquelmarti in https://github.com/kornia/kornia/pull/2185
* Move padding_size to device in pad for boxes by @miquelmarti in https://github.com/kornia/kornia/pull/2197
* Return boxes tensor directly if no boxes by @miquelmarti in https://github.com/kornia/kornia/pull/2196
* Make value an attribute of RandomErasing instances again by @miquelmarti in https://github.com/kornia/kornia/pull/2195
* TensorWrapper bug fix + add __radd__, __rmul__, __rsub__ by @cjpurackal in https://github.com/kornia/kornia/pull/2190
* Fix/repr bug by @neyazbasheer in https://github.com/kornia/kornia/pull/2207
* Replace `assert_allclose` by `assert_close` by @johnnv1 in https://github.com/kornia/kornia/pull/2210
* Fix random crop for keypoints on CUDA device by @johnnv1 in https://github.com/kornia/kornia/pull/2209
* Remove outdated augmentation example by @johnnv1 in https://github.com/kornia/kornia/pull/2206
* Fix CUDA failing tests of same device on `Augmentations` by @johnnv1 in https://github.com/kornia/kornia/pull/2215



## :zap:  Improvements

* add `PadTo` to docs by @johnnv1 in https://github.com/kornia/kornia/pull/2122
* add colormap and `apply_ColorMap` for integer tensor by @johnnv1 in https://github.com/kornia/kornia/pull/1996
* Fix numerical stability for binary focal loss by @zimka in https://github.com/kornia/kornia/pull/2125
* Add RandomGaussianBlur with instance-level gaussian kernel generation by @juliendenize in https://github.com/kornia/kornia/pull/1663
* add transparent pad to `CenterCrop` docs example by @johnnv1 in https://github.com/kornia/kornia/pull/2124
* Ensure support to Python 3.9 and 3.10 by @johnnv1 in https://github.com/kornia/kornia/pull/2025
* improve `TestUpscaleDouble` by @johnnv1 in https://github.com/kornia/kornia/pull/2147
* DataKey: add 'image' as alias of 'input' by @adamjstewart in https://github.com/kornia/kornia/pull/2193
* add `fail-fast:false` as default on tests workflow by @johnnv1 in https://github.com/kornia/kornia/pull/2146
 [enhance] improve flipping and cropping speed by @shijianjian in https://github.com/kornia/kornia/pull/2179
* Replace jit test method  in favor of dynamo in `BaseTester` by @johnnv1 in https://github.com/kornia/kornia/pull/2120
* Small refactor on `filters` module: Dropping JIT support by @johnnv1 in https://github.com/kornia/kornia/pull/2187
* Augmentation Base Refactor by @shijianjian in https://github.com/kornia/kornia/pull/2117


### Deprecation

* move kornia check api to kornia.core.check by @edgarriba in https://github.com/kornia/kornia/pull/2143
* Remove py 3.7 for nightly CI by @johnnv1 in https://github.com/kornia/kornia/pull/2204


## :rocket: [0.6.9] - 2022-12-21
### :new:  New Features

* Feat/randombrightness contrast saturation hue by @duc12111 in https://github.com/kornia/kornia/pull/1955
* Liegroups by @edgarriba in https://github.com/kornia/kornia/pull/1887
* Add sepia by @johnnv1 in https://github.com/kornia/kornia/pull/1947
* Normalize with intrinsics by @ducha-aiki in https://github.com/kornia/kornia/pull/1727
* [feat] liegroup so2 by @cjpurackal in https://github.com/kornia/kornia/pull/1973
* [feat] adjoint for se2, so2 by @cjpurackal in https://github.com/kornia/kornia/pull/2101
* add trans, trans_x, trans_y + minor changes se2 by @cjpurackal in https://github.com/kornia/kornia/pull/2103
* Motion blur by @nitaifingerhut in https://github.com/kornia/kornia/pull/2075
* Add `Hyperplane` and `Ray` API by @edgarriba in https://github.com/kornia/kornia/pull/1963


### :lady_beetle: Bug fixes

* Quaternion pow bug fix (div by zero) by @cjpurackal in https://github.com/kornia/kornia/pull/1946
* fix cuda init by @ducha-aiki in https://github.com/kornia/kornia/pull/1953
* Documentation: proper Sørensen–Dice coefficient by @sergiev in https://github.com/kornia/kornia/pull/1961
* quaternion, so3 and se3 as non batched by @edgarriba in https://github.com/kornia/kornia/pull/1997
* Bump pytest-mypy from 0.10.0 to 0.10.1 by @dependabot in https://github.com/kornia/kornia/pull/2005
* Join the gh-actions for docs by @johnnv1 in https://github.com/kornia/kornia/pull/2003
* [pre-commit.ci] pre-commit suggestions by @pre-commit-ci in https://github.com/kornia/kornia/pull/2010
* So2 bug fix by @cjpurackal in https://github.com/kornia/kornia/pull/2015
* Fix type annotation for torch 1.13.0 by @johnnv1 in https://github.com/kornia/kornia/pull/2023
* Fix an error in `match_smnn` by @anstadnik in https://github.com/kornia/kornia/pull/2020
* Set equal_nan to False in assert_close by @edgarriba in https://github.com/kornia/kornia/pull/1986

## :zap:  Improvements

* minor improvements to So3 by @cjpurackal in https://github.com/kornia/kornia/pull/1966
* Add `TensorWrapper`, `Vector3`, `Scalar` and improvements in `fit_plane` by @edgarriba in https://github.com/kornia/kornia/pull/
* [feat] add vee to so2, se2 by @cjpurackal in https://github.com/kornia/kornia/pull/2091
* Remove deprecated code in `kornia.augmentation` by @johnnv1 in https://github.com/kornia/kornia/pull/2052
* [feat] Implement se2 by @cjpurackal in https://github.com/kornia/kornia/pull/2019
* add quaternion to euler conversion by @edgarriba in https://github.com/kornia/kornia/pull/1994
* use resample instead of mode argument in RandomElasticTransform per default by @JanSellner in https://github.com/kornia/kornia/pull/2017
* replacing .repeat(...) with .expand(...) by @nitaifingerhut in https://github.com/kornia/kornia/pull/2059
* making `RandomGaussianNoise` play nicely on GPU by @nitaifingerhut in https://github.com/kornia/kornia/pull/2050
* None for align_corners arg of resize op with nearest mode by @miquelmarti in https://github.com/kornia/kornia/pull/2049
* facedetector now returns a list of tensors containing the boxes x image by @lferraz in https://github.com/kornia/kornia/pull/2034
* add random for liegroups by @cjpurackal in https://github.com/kornia/kornia/pull/2041
* add rotation and translation classmethods in se3 and so3 by @edgarriba in https://github.com/kornia/kornia/pull/2001
* implement `kornia.geometry.linalg.euclidean_distance` by @edgarriba in https://github.com/kornia/kornia/pull/2000


### Deprecation

* Drop pytorch 1.8 (LTS) support by @johnnv1 in https://github.com/kornia/kornia/pull/2024


## :rocket: [0.6.8] - 2022-10-13
### :new:  New Features

* NeRF Implementation by @YanivHollander in https://github.com/kornia/kornia/pull/1911
* [Feat] Added AugmentationDispatcher by @shijianjian in https://github.com/kornia/kornia/pull/1914
* Add `EdgeDetection` api by @edgarriba in https://github.com/kornia/kornia/pull/1483
* [feat] slerp implementation for Quaternion by @cjpurackal in https://github.com/kornia/kornia/pull/1931
* add laplacian pyramid by @lafith in https://github.com/kornia/kornia/pull/1816
* Added homography from line segment correspondences by @ducha-aiki in https://github.com/kornia/kornia/pull/1851
* [feat] Added Jigsaw Augmentation by @shijianjian in https://github.com/kornia/kornia/pull/1852

### :lady_beetle: Bug fixes

* Fix svdvals usage by @ducha-aiki in https://github.com/kornia/kornia/pull/1926
* fix shift_rgb stack dimension by @nmichlo in https://github.com/kornia/kornia/pull/1930
* Update kernels.py by @farhankhot in https://github.com/kornia/kornia/pull/1940
* Quaternion.norm bug fix by @cjpurackal in https://github.com/kornia/kornia/pull/1935
* Fix quaternion doctests by @edgarriba in https://github.com/kornia/kornia/pull/1943
* Remove unnecessary CI jobs by @johnnv1 in https://github.com/kornia/kornia/pull/1933
* fix cuda tests failing by @ducha-aiki in https://github.com/kornia/kornia/pull/1941
* No crash in local feature matching if empty tensor output by @ducha-aiki in https://github.com/kornia/kornia/pull/1890


### :zap:  Improvements

* RANSAC improvements by @ducha-aiki in https://github.com/kornia/kornia/pull/1435
* Make AdaLAM output match confidence by @ducha-aiki in https://github.com/kornia/kornia/pull/1862
* Enlargen LoFTR positional encoding map if large images are input by @georg-bn in https://github.com/kornia/kornia/pull/1853


## :rocket: [0.6.7] - 2022-08-30
### :new:  New Features

* Added FGINN matching by @ducha-aiki in https://github.com/kornia/kornia/pull/1813
* Added SOLD2 by @rpautrat  https://github.com/kornia/kornia/pull/1507 https://github.com/kornia/kornia/pull/1844
* edge aware blur2d by @nitaifingerhut in https://github.com/kornia/kornia/pull/1822
* Adds conversions between graphics and vision coordinate frames by @ducha-aiki in https://github.com/kornia/kornia/pull/1823
* Add Quaternion API by @edgarriba in https://github.com/kornia/kornia/pull/1801
* AdaLAM match filtering by @ducha-aiki in https://github.com/kornia/kornia/pull/1831
* Init Mosaic Augmentation by @shijianjian in https://github.com/kornia/kornia/pull/1713


### :lady_beetle: Bug fixes

* fix tests float16 module losses by @MrShevan in https://github.com/kornia/kornia/pull/1809

### :zap:  Improvements

* Allowing more than 3/4 dims for `total_variation` + adding `reduction` by @nitaifingerhut in https://github.com/kornia/kornia/pull/1815


## :rocket: [0.6.6] -  - 2022-07-16

### :new:  New Features

* Add `ParametrizedLine`  and `fit_line` by @edgarriba in https://github.com/kornia/kornia/pull/1794
* Implement `project` and `unproject` in `PinholeCamera` by @YanivHollander in https://github.com/kornia/kornia/pull/1729
* adding `rgb_to_y` by @nitaifingerhut in https://github.com/kornia/kornia/pull/1734
* add `KORNIA_CHECK_SAME_DEVICES` by @MrShevan in https://github.com/kornia/kornia/pull/1788


### Deprecation

* deprecate `filter2D` `filter3D` api by @edgarriba in https://github.com/kornia/kornia/pull/1725


### :lady_beetle: Bug fixes

* fixes for half precision in imgwarp by @edgarriba in https://github.com/kornia/kornia/pull/1723
* Fix transforms for empty boxes and keypoints inputs by @hal-314 in https://github.com/kornia/kornia/pull/1741
* fixing doctest in pinhole by @edgarriba in https://github.com/kornia/kornia/pull/1743
* Fix/crop transforms by @hal-314 in https://github.com/kornia/kornia/pull/1739
* Fix Boxes.from_tensor(boxes, mode="vertices") by @hal-314 in https://github.com/kornia/kornia/pull/1740
* fix typing callable in load storage by @edgarriba in https://github.com/kornia/kornia/pull/1768
* Fix bug preventing sample wise augmentations by @ashnair1 in https://github.com/kornia/kornia/pull/1761
* Refactor and add tests in `get_perspective_transform` by @edgarriba in https://github.com/kornia/kornia/pull/1767


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
- `pos_weight` param to focal loss (#1744)

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
