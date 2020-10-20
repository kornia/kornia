# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
