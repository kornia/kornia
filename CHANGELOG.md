# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pending changes live in [changelog.d](changelog.d/README.md).
Run `pixi run changelog-preview` to read the assembled notes.

****

<!-- towncrier release notes start -->

## [0.9.0] - 2026-10-23

### Added

- `kornia.metrics.kitti_d1_error`, the outlier ratio reported by the KITTI 2015 stereo benchmark.
  A pixel counts as an outlier only when its absolute disparity error exceeds `abs_threshold`
  **and** its relative error exceeds `rel_threshold`, both strictly, with the ground truth as the
  relative denominator. That conjunction is what separates it from `mean_bad_pixel_error`, which
  applies the absolute criterion alone and so penalises large disparities for errors that are small
  next to their magnitude. A batch is pooled over every valid pixel, as the devkit accumulates it,
  rather than averaged per image. Closes #3758. ([#3931](https://github.com/kornia/kornia/pull/3931))
- `PadTo` takes `crop_if_exceeds`. The default `True` keeps the existing behavior: an input larger than `size` on an axis is cropped to `size` there. With `crop_if_exceeds=False` that axis passes through unchanged, so the output is at least `size` on each axis and boxes and keypoints stay inside the image. Closes #3518. ([#3964](https://github.com/kornia/kornia/pull/3964))
- Added opt-in `kornia.contrib.augmentation_audit` reports with executed geometric provenance,
  native parameter snapshots, out-of-frame diagnostics and actual keypoint/box round-trip
  measurements, including explicit unsupported transforms and JSON export. ([#4465](https://github.com/kornia/kornia/pull/4465))
- Relevant camera and image-coordinate API docstrings now link directly to the camera and general conventions guides. ([#4501](https://github.com/kornia/kornia/pull/4501))
- Document per-class coordinate, sampling, crop, resize, padding, and non-rigid warp conventions for 2D geometric augmentations, with executable tests for current behavior and known limitations. Clarify sampler migration after module and container moves. ([#4538](https://github.com/kornia/kornia/pull/4538))
- Document per-class value-range, parameter-meaning, filter, noise, weather, illumination and normalization conventions for the 2D intensity augmentations, with executable tests for current behavior and issue-linked pins for the known limitations. Rewrite the stale sampler-migration citations in the 2D augmentation bases. ([#4566](https://github.com/kornia/kornia/pull/4566))
- Document the conventions of mix and 3D augmentations and automatic augmentation policies, with executable tests for their sampling, coordinate, label, and output contracts. ([#4634](https://github.com/kornia/kornia/pull/4634))
- `SIFTFeature` and `SIFTFeatureScaleSpace` now accept `descriptor_backend="pyramid"` for fast extraction at their sparse detections. `SIFTDescriptorFromPyramid` describes supplied LAFs and can share an extraction pyramid between orientation assignment and description.

  The scale-space preset has a dedicated backend that reuses the detector Gaussian pyramid and retained octave/layer assignments, computing gradients once per used level. It preserves sparse detections and integrates rotated SIFT histograms directly from gradient fields.

  The optimized scale-space path also specializes DoG detection: sparse refinement, exact octave coordinates, and top-K response ranking without contrast or edge rejection. The default patch backend remains available.

  CPU Gaussian filtering avoids convolution work buffers; sparse neighbourhood gathers, two-bin angular voting, and device-specific descriptor chunks reduce extraction time without reducing integration support or the requested feature budget. Compiled and eager pyramid backends share checkpoint keys and support full-module serialization.

  CPU inference builds Gaussian volumes directly in their final storage and reuses sampling coordinates; gradient atlases also avoid a full transpose copy. Spatial ties are rejected before bounded sparse gathers. Empty detections and descriptors retain zero gradients, and half-precision masks preserve their suppression semantics.

  CUDA uses packed sparse refinement and separable descriptor histogram pooling to reduce kernel launches and extraction time while retaining bounded descriptor chunks. CPU and MPS keep their existing refinement paths.

  Shared `conv_quad_interp3d` and `iterative_quad_interp3d` now group finite differences and Cramer determinants on CUDA float32/float64, also accelerating the default patch-based `SIFTFeatureScaleSpace` pipeline. CPU, MPS and reduced-precision inputs retain their scalar paths. ([#4638](https://github.com/kornia/kornia/pull/4638))
- `depth_from_disparity` accepts a per-batch-element `(B,)` `baseline` and `focal`, so a batch of stereo pairs from
  rigs with different baselines or focal lengths no longer has to be split and processed one sample at a time.
  Element `b` pairs with `disparity[b]`, which needs a disparity of rank 3 or more with a leading size of `B`; a
  mismatched size still raises `ShapeError`. The shared forms, a Python number, a scalar tensor or a `(1,)` tensor,
  behave exactly as before, and the two may be mixed. ([#4648](https://github.com/kornia/kornia/pull/4648))
- Added `kornia.contrib.connected_components_union_find`, an exact, pure PyTorch 8-connected component labeling alternative using 2x2 block union-find. It converges without a user-selected iteration budget and returns integer labels for boolean, integer and floating-point masks, including half precision. ([#4666](https://github.com/kornia/kornia/pull/4666))
- Document the conventions of `RandomTransplantation` and `RandomTransplantation3D` — mask-driven donor selection, the `(i - 1) mod B` donor rule, label sampling and `excluded_labels`, the replay contract, and the data-key and dtype rules — with executable tests, and export both names from `kornia.augmentation.__all__`. ([#4695](https://github.com/kornia/kornia/pull/4695))
- Add opt-in numerical reconstruction masks to both stereo disparity reprojection entry points with `return_mask=True`, preserving existing coordinates, precision and gradients. The mask checks finite homogeneous components, the existing absolute W cutoff and finite returned XYZ; it does not certify matches, front-of-camera eligibility or visibility. ([#4701](https://github.com/kornia/kornia/pull/4701))
- `ColorJiggle` now accepts a fixed `order` for deterministic application and full-graph compilation on RGB inputs. ([#4717](https://github.com/kornia/kornia/pull/4717))
- `kornia.morphology` gains `engine="shift"`: a running max or min over the shifted views of the padded image. For finite inputs its output equals `engine="unfold"` (`torch.equal`; only the sign of a zero can differ), `torch.compile` fuses it into a single kernel, and forward-only its peak memory is flat in the kernel area instead of growing with it (69 MiB of peak RSS growth against 1380 MiB for `"unfold"` with a 15x15 kernel on 8x3x256x256 float32 on CPU). Under autograd it saves two output-sized intermediates per kernel position, so `engine="auto"` keeps `"unfold"` for the float32/float64 CPU backward path. ([#4728](https://github.com/kornia/kornia/pull/4728), [#4996](https://github.com/kornia/kornia/pull/4996))
- Document the conventions of `kornia.morphology` — the kernel reflection in `dilation` and its absence in
  `erosion`, the membership-mask `kernel` and additive `structuring_element`, the `[row, col]` `origin` and its
  even-size default, and the five `border_type` values with their scipy / scikit-image names (and OpenCV's for
  `geodesic` and `reflect`) — as Convention blocks on the seven public functions and a new section on the Conventions &
  Pitfalls page, with executable tests for the kornia-side claims and for the `max_val` sentinel, non-float
  operand and convolution-engine dtype warts. ([#4743](https://github.com/kornia/kornia/pull/4743))
- The `kornia.core` API page documents the six exception classes in `kornia.core.exceptions` (`BaseError`, `ShapeError`, `TypeCheckError`, `ValueCheckError`, `DeviceError`, `ImageError`), so references to them from other pages now link. ([#4798](https://github.com/kornia/kornia/pull/4798))
- `camtoworld_to_worldtocam_Rt` and `worldtocam_to_camtoworld_Rt` accept `check_rotation=True`, which raises `ValueError` when `R` is not a rotation matrix instead of silently returning a wrong pose. ([#4817](https://github.com/kornia/kornia/pull/4817))
- `rotation_matrix_to_axis_angle`, `rotation_matrix_to_quaternion`, `Quaternion.from_matrix`,
  `So3.from_matrix` and `Se3.from_matrix` take `check_rotation` (default `False`). With
  `check_rotation=True` they raise a `ValueError` unless every input is a rotation:
  `max|R @ R^T - I|` within 100 eps of the input dtype (16 eps for float16 and bfloat16) and
  `det(R) > 0`. Without it the behaviour is unchanged, so a reflection such as `diag(-1, 1, 1)`
  is still reported as the identity rotation `[0., 0., 0.]`, and `Se3.from_matrix` still accepts
  a reflection block. (#4773) ([#4840](https://github.com/kornia/kornia/pull/4840))
- Convention blocks for the two-view F/E contract (`find_fundamental`, `find_essential`, the epipolar metrics, `decompose_essential_matrix` and their siblings), a "Two-view geometry" section on the Conventions & Pitfalls page with the OpenCV mapping, and 28 convention and wart pins; the warts are documented and linked (#2198, #4831, #4855, #4862, #4874–#4881, #4883, #4884), not fixed. ([#4887](https://github.com/kornia/kornia/pull/4887))
- Convention blocks for the projection and structure functions of `kornia.geometry.epipolar` (`projection_from_KRt`, `KRt_from_projection`, `projections_from_fundamental`, `depth_from_point`, `scale_intrinsics`, `intrinsics_like`, `random_intrinsics`, `triangulate_points`, `generate_scene`), three functions added to the rendered API reference, and 16 convention and wart pins; the #4263 wart is documented and linked, not fixed; #4863, #4864, #4865 and #4900 are fixed by #4901. ([#4889](https://github.com/kornia/kornia/pull/4889))
- Convention blocks for homography estimation (`find_homography_dlt` and its iterated and line-segment variants, the transfer errors), `RANSAC` and the polynomial and homogeneous solvers (`solve_quadratic`, `solve_cubic`, `solve_quartic`, `determinant_to_polynomial`, `null_vector_3x4`), with convention and wart pins; `solve_quartic` is added to the rendered API reference. The documented defects are linked to their issues (#4833, #4866–#4870, #4872–#4874, #4881, #4890, #4905) and not fixed here. ([#4892](https://github.com/kornia/kornia/pull/4892))
- `RANSAC(prosac_sampling=True)` now draws progressive samples from correspondences sorted best-first, following Chum
  and Matas (CVPR 2005); the flag used to be accepted and ignored. The schedule advances per sampled set inside each
  batch, and the sampler runs the full `batch_size * max_iter` budget. `RANSAC` also accepts `lo_sample_size`, which
  fits `max_lo_iters` random inlier subsets of that size in one solver batch before a final full-inlier refit. ([#4902](https://github.com/kornia/kornia/pull/4902))
- `normalize_points` accepts optional per-point `weights` of shape `(B, N)`: the centroid and the mean radius are
  weighted, negative weights count as zero, and a batch element whose weights are all zero uses the unweighted
  statistics. ([#4908](https://github.com/kornia/kornia/pull/4908))
- Convention blocks for `Quaternion`, `average_quaternions` and `So3` (storage, product order, principal `log`, Jacobian sides), a "Rotations and rigid motions" section on the Conventions & Pitfalls page with the scipy / Sophus / Eigen mapping, and 19 convention and wart pins; the warts are documented and linked (#3950, #3953, #4923, #4942), not fixed. ([#4976](https://github.com/kornia/kornia/pull/4976))
- Convention blocks for `So2`, `Se2` and `Se3` (tangent order, direction, `log` range, adjoint and matrix layouts, composition order) and 25 convention and wart pins; the warts are documented and linked (#4923, #4929, #4931, #4932, #4942), not fixed. ([#4977](https://github.com/kornia/kornia/pull/4977))
- Convention blocks for `relative_transformation` and its linalg siblings (frame naming, rigid-only inputs, the contrast with `relative_camera_motion`) and for `NamedPose`, an API page for `kornia.geometry.pose`, and 12 convention and wart pins; the warts are documented and linked (#4881, #4937), not fixed. ([#4978](https://github.com/kornia/kornia/pull/4978))
- `RANSAC` gains `local_optimization` (`"lm"` or `"dlt"`), `refine_iters` for the final Levenberg-Marquardt refinement, the `"fundamental_8pt"` model type, and an `offset` argument to `RANSAC.sample` that continues the PROSAC schedule across batches of different sizes. The new pipeline normalizes the correspondences once per call, solves minimal samples through a batched LU null space and scores them with one matrix product per batch; on PhotoTourism fundamental matrices and HEB homographies it is more accurate than the previous defaults and several times faster on CUDA. `sample_is_valid_for_homography` no longer copies index tensors to the device on every call. The minimal solvers share their LU null space with the public estimators: `find_fundamental(method="7POINT")`, the eight-correspondence unweighted case of `find_fundamental`, and the four-correspondence `find_homography_dlt(solver="lu")` replace `eigh`, QR or SVD with one batched LU factorization, which is several times faster on CPU and now runs in float16 and bfloat16; the seven-point cubic is expanded without inverting the null-space basis. `sampson_epipolar_distance` and `oneway_transfer_error` (and so `symmetric_transfer_error`) score several matrices on one set of correspondences with one or two matrix products, faster on CPU and CUDA for RANSAC-sized batches, and `find_fundamental`'s eight-point solver removes the smallest singular value in closed form instead of with an SVD for batches of 128 or more on CPU and 512 or more on other devices, about twice as fast for 2048 samples. ([#5031](https://github.com/kornia/kornia/pull/5031))
- `ImageRegistrator.warp_dst_into_src`, the correctly spelled inverse of `warp_src_into_dst`; its docstring now names the argument it warps, `dst_img`. The misspelled `warp_dst_inro_src` still works as a deprecated alias that emits a `DeprecationWarning`. ([#5049](https://github.com/kornia/kornia/pull/5049))
- Convention blocks for `create_meshgrid`, `create_meshgrid3d` and `kornia.geometry.subpix` (the `(x, y)` / `(d, x, y)` coordinate orders, corner-aligned normalization, the `grid_sample` reorder for 3D grids, the conv soft-argmax windows, the quadratic refiners' rejection rules and strict NMS), the matching additions to the Conventions & Pitfalls page, and 14 convention and wart pins; the warts are documented and linked (#4503, #5020, #5065, #5066), not fixed. ([#5068](https://github.com/kornia/kornia/pull/5068))
- Convention blocks for `Keypoints`, `Vector3` and the PLY writers (`(x, y)` pixel coordinates, homogeneous transformation, copy / in-place / write-through aliasing, the `double` `x y z` PLY layout), an API page for `kornia.geometry.vector`, and 13 convention and wart pins; the warts are documented and linked (#5022, #5023), not fixed. ([#5069](https://github.com/kornia/kornia/pull/5069))
- Convention blocks for `ParametrizedLine`, `fit_line`, `Hyperplane`, `fit_plane` and `ImageRegistrator` (unnormalized line directions, the `n . x + d = 0` plane and the orientation of `through`, the built-in destination-to-source registration models in normalized coordinates), `fit_plane` and the `Hyperplane` methods on the line API page, and 13 convention and wart pins; the wart is documented and linked (#4923), not fixed. ([#5070](https://github.com/kornia/kornia/pull/5070))
- `kornia.image.draw_rectangle` now accepts `line_width` to draw outlines thicker than one pixel. The outline grows
  inward from each edge, so it stays inside the rectangle, and the part of it outside the image is not drawn. ([#5093](https://github.com/kornia/kornia/pull/5093))
- `RANSAC` gains opt-in `degensac=True` (off by default), DEGENSAC's dominant-plane recovery for seven-point fundamental matrices (Chum, Werner and Matas, CVPR 2005), following Chum's implementation in pydegensac: record-setting seven-point models are tested for five or more correspondences related by one homography, and a degenerate one hands that homography to a plane-and-parallax search over the correspondences off the plane. The new `kornia.geometry.homography.sampson_homography_distance` returns the Sampson distance of correspondences to homographies, the first-order geometric error over both images (Hartley and Zisserman, section 4.2.6): unlike `oneway_transfer_error`, it accounts for noise in both images and does not depend on the scale of `H`, and an exact correspondence scores 0. ([#5099](https://github.com/kornia/kornia/pull/5099))
- `RANSAC(compile=True)` runs the whole `local_optimization="lm"` estimation (homography, fundamental and essential
  matrices) as one `torch.compile` graph on CPU and CUDA with torch 2.14 or later. The graph is traced once per configuration and does not
  depend on the number of correspondences, the threshold, the confidence or the sample budget; it is saved next to
  inductor's cache, so a later process loads it instead of compiling. Seeded compiled calls are reproducible but use
  their own random stream without changing the global CPU or CUDA generators. Ordinary and inference-mode calls
  share the compiled artifact. Eager results are unchanged.

  Compiled calls normalize autocast, distinguish thread/default-dtype/default-device settings in the artifact cache,
  and draw their samples from a counter-based hash of the seed, so concurrent seeded calls preserve the global RNG. Large explicit CPU batches
  split to bound scoring memory while retaining the sample budget. Zero-weight singular correspondences no longer
  prevent LM refinement; artifact writes are atomic. ([#5117](https://github.com/kornia/kornia/pull/5117))
- `convert_sift_descriptor_layout` reorders SIFT descriptors between kornia's layout and OpenCV's, and `SIFTFeature` and `SIFTFeatureScaleSpace` take `descriptor_layout="opencv"` to return OpenCV's layout directly. `LightGlueMatcher("sift")` expects OpenCV's layout, which kornia's SIFT descriptors do not use by default, so request it or convert before matching: on Oxford Graf 1→4, `SIFTFeature` descriptors matched in kornia's layout gave 2 correct matches, and 497 in OpenCV's. ([#5145](https://github.com/kornia/kornia/pull/5145))
- Convention blocks for `filter2d`, `filter2d_separable`, `filter3d`, `fft_conv` and the kernel builders: correlation vs `behaviour='conv'`, the `border_type` vocabulary, the `(k - 1) // 2` anchor of an even kernel, kernel batch and normalisation rules, the Rodrigues axis-angle vector used by `get_motion_kernel3d`, and the orientation, sign and normalisation of every derivative kernel. A new Filtering section on the Conventions page maps the border modes, the correlation rule and the even-kernel anchor to scipy and OpenCV. `fft_conv` and ten kernel builders are added to the API reference. ([#5230](https://github.com/kornia/kornia/pull/5230))
- `correlate2d`, `convolve2d`, `correlate3d` and `convolve3d` are now listed in `kornia.filters.__all__`, so `from kornia.filters import *` brings them, and appear on the Filtering API page and in the API-surface inventory. They were already importable from `kornia.filters` by name, and equal `filter2d` / `filter3d` with `behaviour='corr'` / `'conv'`. The two 2-D docstrings now state the output shape for `padding='valid'`, and each of the four has an example that shows how correlation and convolution differ. ([#5249](https://github.com/kornia/kornia/pull/5249))
- `kornia.morphology.reconstruction`: grayscale morphological reconstruction by dilation or erosion, matching `skimage.morphology.reconstruction`. ([#5257](https://github.com/kornia/kornia/pull/5257))
- Document color-space and enhancement conventions, correct channel units and API contracts, and add executable convention and known-defect tests. ([#5319](https://github.com/kornia/kornia/pull/5319))
- Convention blocks for `spatial_gradient`, `spatial_gradient3d`, `sobel`, `laplacian`, `canny`, `in_range` and `otsu_threshold`: the `(dx, dy)` output layout and default normalisation of the derivative filters, the Laplacian kernel's sign and scale, the unnormalized Sobel magnitude that `canny`'s thresholds are compared with and its tie rule, `in_range`'s inclusive bounds and accepted bound shapes, and the per-plane Otsu threshold read from the upper edge of the split bin. The Filtering section of the Conventions page maps the derivative scales and `canny`'s thresholds to OpenCV and scikit-image. ([#5427](https://github.com/kornia/kornia/pull/5427))
- Convention blocks for the blurring operators of `kornia.filters` (`gaussian_blur2d`, `box_blur`, `median_blur`, `bilateral_blur`, `joint_bilateral_blur`, `guided_blur`, `motion_blur`, `motion_blur3d`, `unsharp_mask`, `blur_pool2d`, `max_blur_pool2d`, `edge_aware_blur_pool2d` and their module classes): `kernel_size` and `sigma` given rows first, per-sample tensor sigmas, the bilateral and guided filters' parameter units and argument order, the motion-blur angle and streak weighting, the blur pools' sampling grid and padding. The Filtering section of the Conventions page relates them to OpenCV, scipy, scikit-image and the antialiased-cnns `BlurPool`. `edge_aware_blur_pool2d`, `motion_blur3d`, `EdgeAwareBlurPool2D`, `MotionBlur3D` and `StableDiffusionDissolving` are added to the API reference. ([#5433](https://github.com/kornia/kornia/pull/5433))
- `kornia.feature.match_smnn_batched` matches independent descriptor pairs with batched distance computation, packed match indices, and optional validity masks for padded inputs. ([#5558](https://github.com/kornia/kornia/pull/5558))
- Convention blocks for the image-quality and regression losses and metrics (`ssim`, `ssim3d`, `psnr`, the SSIM / MS-SSIM / PSNR losses, the Charbonnier / Cauchy / Geman–McClure / Welsch losses, total variation, inverse-depth smoothness and the KL / JS divergences), with `test_convention_*` pins and a new "Losses and metrics" section on the conventions page. ([#5566](https://github.com/kornia/kornia/pull/5566))
- Convention blocks for the mutual-information losses (`mutual_information_loss`, its normalized, 2-D, 3-D and `FromRef` module forms, and `MIKernel`), with `test_convention_*` pins and a new Registration API page that lists all 13 mutual-information exports. ([#5632](https://github.com/kornia/kornia/pull/5632))
- Convention blocks for the dense-prediction losses (`focal_loss`, `binary_focal_loss_with_logits`, `dice_loss`, `tversky_loss`, the Lovász and Hausdorff losses and `one_hot`), with `test_convention_*` pins and a "Dense-prediction losses" sub-section on the conventions page. ([#5634](https://github.com/kornia/kornia/pull/5634))
- Convention blocks for the task metrics (classification, segmentation, detection, optical flow, disparity and pose), with `test_convention_*` pins and a "Task metrics" sub-section on the conventions page that tabulates each metric's output, pooling and scale. ([#5635](https://github.com/kornia/kornia/pull/5635))
- The docstrings of `shear`, `Shear`, `Affine`, `get_affine_matrix2d`, `get_shear_matrix2d` and `RandomAffine` now say what their shear arguments mean: raw factors about pixel (0, 0) for `shear` and `Shear`, and angles in radians about the centre, sheared by their negative tangent, for `Affine`, the two matrix helpers and `RandomAffine` (in degrees there). That two entry points read `shear` differently is a known defect (#5661). ([#5682](https://github.com/kornia/kornia/pull/5682))
- `DoGHardNet` and `DoGAffNetHardNet` return LAFs and HardNet descriptors that `LightGlueMatcher("doghardnet")` and `LightGlueMatcher("dog_affnet_hardnet")` take without rescaling. They detect with the DoG detector of the default `"patch"` backend of `SIFTFeatureScaleSpace`, `DoGAffNetHardNet` adding AffNet shapes, estimate the orientation at the detection scale, and return and describe LAFs twice the detection frames: 12 sigma, the scale those LightGlue weights were trained with. The `LightGlueMatcher` documentation states that scale. ([#5727](https://github.com/kornia/kornia/pull/5727))
- Document an explicit scale-space pyramid SIFT recipe, distinguish multiresolution detection
  from scale-space detection, and explain the patch, dense and supplied-frame descriptor interfaces.
  Document the repaired multiresolution pyramid orientation and its remaining validation and
  possible-deprecation criteria without changing defaults, adding runtime warnings or removing APIs. ([#5750](https://github.com/kornia/kornia/pull/5750))
- `kornia.image.ColorSpace` is exported, so a `PixelFormat` or a colour space for `Image.from_numpy` no longer needs an
  import from `kornia.image.base`. `draw_point2d` and `draw_line` accept a single-value colour, of shape `(1,)` or 0-d,
  and write it to every channel, as `draw_rectangle` and `draw_convex_polygon` already did; they used to raise for it on
  an image with more than one channel. OpenCV's and Pillow's scalar colours fill only the first channel. ([#5813](https://github.com/kornia/kornia/pull/5813))
- The ONNX guide and the landing page now export with the `torch.export`-based exporter (`dynamo=True`, with
  `dynamic_shapes`) instead of the deprecated TorchScript-based one at opset 17, and show the
  `auto_ir_version_conversion=True` flag that `ONNXSequential` needs to chain such an export with the pre-exported Hub
  operators. ([#5818](https://github.com/kornia/kornia/pull/5818))
- The ONNX, torch.compile and torch.export page is now presented as a dated survey rather than a support statement:
  it is titled a survey, opens with a note that it records one measurement that Kornia's tests check only in part,
  and reports *passed / probed* instead of *supported / probed*. The ONNX guide no longer quotes the share of
  operators that export. ([#5820](https://github.com/kornia/kornia/pull/5820))
- The `kornia.io` and `kornia.image` API documentation states their conventions: the channels, dtype and range of each
  `load_image` type, the dtypes and channel counts each `write_image` format stores, when the NumPy converters share
  memory, the pixel frame, rounding and clipping of the drawing functions, the value ranges of the terminal
  printers, and the supported dtypes for `Image.to_gray()`. The conventions page gains a *Files and arrays* section,
  with the load types and a table mapping OpenCV's, Pillow's and torchvision's readers, writers and converters onto
  kornia's, and a *Drawing* section. One known defect is documented: `write_image` raises a bare `Exception` and
  leaves an empty or truncated file for an image with zero height or width (#5770). ([#5823](https://github.com/kornia/kornia/pull/5823))
- Document the conventions of the local affine frame (LAF) and patch-extraction API (`kornia.feature.laf`, `get_laf_descriptors`, `LAFDescriptor` and `KORNIA_CHECK_LAF`) as Convention blocks and pointers, and add a "Local features" section to the Conventions & Pitfalls page: the `[A | c]` frame in pixels, scale and orientation, the patch a LAF samples, detector padding, and the conversion from OpenCV keypoints. ([#5827](https://github.com/kornia/kornia/pull/5827))
- Document the conventions of the `kornia.feature` response functions, detectors, orientation and affine-shape estimators as Convention blocks and pointers: the sign and intensity scaling of each response, the candidates, frame and padding of `ScaleSpaceDetector`, the level-pixel frame of `MultiResolutionDetector`, the input range of `KeyNet`, the orientation sign, and the layout of the `PatchAffineShapeEstimator` output. ([#5835](https://github.com/kornia/kornia/pull/5835))
- Document the conventions of the `kornia.feature` patch descriptors (`SIFTDescriptor`, `DenseSIFTDescriptor`, `SIFTDescriptorFromPyramid`, `HardNet`, `HardNet8`, `SOSNet`, `TFeat`, `HyNet`, `FilterResponseNorm2d`, `TLU` and `MKDDescriptor`) as Convention blocks and pointers: the patch shape and sampling, which intensity changes leave a descriptor unchanged, the output norm and width, SIFT clipping and RootSIFT, and the dense and pyramid SIFT cell geometry. ([#5836](https://github.com/kornia/kornia/pull/5836))
- Document the conventions of the `kornia.feature` matchers (`match_nn`, `match_mnn`, `match_snn`, `match_smnn`, `match_smnn_batched`, `match_fginn`, `match_adalam`, `DescriptorMatcher`, `GeometryAwareDescriptorMatcher`, `LocalFeatureMatcher`, `LightGlueMatcher`, `LightGlue`, `OnnxLightGlue` and `LoFTR`) as Convention blocks and pointers: what the returned values and index pairs hold for each matcher, the inclusive ratio thresholds and a given distance matrix, the keypoints, image size and extra inputs each LightGlue feature type takes, the dtype of LightGlue's matching scores, the LoFTR keypoint frame, and the known defects of `match_adalam`, `LightGlue` and `OnnxLightGlue`. ([#5856](https://github.com/kornia/kornia/pull/5856))
- `LoFTR.extract_features(image)` returns the features of an image that do not depend on the image it is matched with
  (the backbone's coarse and fine maps, the coarse one position-encoded), and `LoFTR.forward` accepts them as
  `features0` / `features1` instead of `image0` / `image1`, on either side or both. Matching one image against many no
  longer runs the backbone for every pair. For images of different sizes the matches are bitwise the same as when
  passing the images; `forward` still runs two images of the same size through the backbone together.
  All image and feature input combinations reject unequal batch sizes with `ValueError`. ([#5880](https://github.com/kornia/kornia/pull/5880))
- Document the conventions of the `kornia.feature` local-feature presets and learned models (`LocalFeature`, `SIFTFeature`, `SIFTFeatureScaleSpace`, the AffNet/HardNet/KeyNet presets, `DISK`, `ALIKED`, `XFeat`, `XFeatModel`, `InterpolateSparse2d`, `DeDoDe`, `SOLD2`, `SOLD2_detector` and `DeFMO`) as Convention blocks and pointers: the input range and layout each model reads, the keypoint frame, order and dtype each returns, how DISK's `n` and ALIKED's top-k and threshold modes select keypoints and where that departs from the reference implementations, which dense descriptors are normalised, the padded slots of `LocalFeature` and why they must be dropped before matching, and the known defects of `SIFTFeature`, `ALIKED`, `SOLD2` and `InterpolateSparse2d`. ([#5899](https://github.com/kornia/kornia/pull/5899))
- `kornia.geometry.conversions.denormalize_homography3d`, the inverse of `normalize_homography3d`, and
  `kornia.geometry.conversions.ColmapQTVecs_to_ARKitQTVecs`, the reverse of `ARKitQTVecs_to_ColmapQTVecs`, so both
  conversions can now be round-tripped through the public API. ([#5938](https://github.com/kornia/kornia/pull/5938))
- Add weighted `kornia.geometry.find_affine` estimation and `RANSAC(model_type="affine")` for full 2D affine transforms. AdaLAM shares the linear solver while retaining its seed-constrained neighborhood RANSAC. Affine fitting and scoring retain at least float32 arithmetic under autocast. ([#5940](https://github.com/kornia/kornia/pull/5940))
- `kornia.tracking` and the numeric and tooling exports of `kornia.contrib` document their conventions in Convention blocks, the contrib page lists `histogram_matching` and `interp` and maps the connected-component connectivity onto SciPy, OpenCV and scikit-image, and the known defect links its issue (#5956). ([#5978](https://github.com/kornia/kornia/pull/5978))
- `kornia.augmentation.CenterCrop`, `RandomCrop` and `RandomResizedCrop` support `inverse` in their default `cropping_mode="slice"`: images and masks, and for `RandomCrop` also boxes and keypoints, are warped back onto the input canvas with the inverse of the recorded `transform_matrix`, as in resample mode. Cropped-away content is not restored. These calls used to raise `NotImplementedError`, except that `CenterCrop` and `RandomResizedCrop` returned a tensor for an empty batch or one the crop was not applied to. ([#5984](https://github.com/kornia/kornia/pull/5984))
- `kornia.models.qwen25.Qwen25VisionTransformer` and `Qwen25VisionConfig`, the Qwen2.5-VL vision tower: a
  temporal patch embedding, a 2D rotary position embedding, RMS-normalized blocks with gated MLPs, windowed
  attention with full attention in selected blocks, and the 2 x 2 patch merger. `from_pretrained` strictly
  loads the `visual.*` tensors of a Qwen2.5-VL checkpoint from a local file or directory. On the released
  3B weights it matches Hugging Face `transformers` v4.51.0 on the same preprocessed patches.
  `kornia.core.load_safetensors` gains an `include` predicate that copies only the selected tensors out of
  the file, so a vision tower can be read from a shard it shares with a language model. Closes #6003. ([#6025](https://github.com/kornia/kornia/pull/6025))
- `SigLip2Model.forward` takes a `pixel_attention_mask` for the vision tower, as transformers' `Siglip2Model.forward` does: it reaches `get_image_features`, so the image embeddings, the logits and the `return_loss` loss leave out the masked patches. ([#6051](https://github.com/kornia/kornia/pull/6051))
- `KORNIA_CHECK_SAME_SIZE` checks that two tensors have the same size along a dimension, optionally accepting a size of 1 for the first, and `KORNIA_CHECK_SIZES` checks a tensor against sizes computed at run time, with `None` for a dimension of any size. Both build their error message only on failure. The `msg` of every `KORNIA_CHECK*` helper can also be a function without arguments, called only when the check fails: `KORNIA_CHECK(condition, lambda: f"...")`. ([#6063](https://github.com/kornia/kornia/pull/6063))
- `kornia.filters.dexined.DexiNed` is now the same class as `kornia.models.dexined.DexiNed`, which that module re-exports with its building blocks, so it gains `to_onnx` and its ONNX declarations. A whole model pickled from `kornia.filters.dexined` still loads, and a newly pickled one is stored under `kornia.models.dexined`. ([#6107](https://github.com/kornia/kornia/pull/6107))
- Added `force_download=True` to `CachedDownloader` and the ONNX loaders, including the Hugging Face community loader and `DepthAnythingONNXBuilder`, to refresh cached models, external weights, and preprocessing configs. Failed downloads preserve the previous cached file. Errors when loading cached models or parsing configs now suggest the option. ([#6139](https://github.com/kornia/kornia/pull/6139))
- The ONNX, torch.compile and torch.export page now opens with "What Kornia promises": the stable-core configurations
  that export with the dynamo ONNX exporter at opset 18 and match eager in onnxruntime, and the ones that compile with
  `torch.compile(..., fullgraph=True)` and match eager, for inference at static shapes, checked in CI on torch 2.9 and
  2.14, with the configurations left out, the known failures and their tracking issues; the survey tables mark every
  promised configuration. `nms` and `draw_rectangle` document how `torch.compile` handles them, and are left out of the
  compile promise (#6194). ([#6233](https://github.com/kornia/kornia/pull/6233))
- API reference entries now carry a **Try in browser** badge that links a function, its
  `nn.Module` counterpart or a browser model to its interactive page on the kornia.org
  playground, so readers can run the operator on a sample image without installing anything.
  The set of linked APIs is a snapshot of the playground in `docs/source/_playground_links.json`,
  refreshed by `docs/generate_playground_links.py`. (#4333)
- An **Adoption** page (`community/adoption`) listing the most-starred GitHub repositories and
  packages that depend on kornia, rendered at build time from `docs/source/_data/dependents.json`;
  `docs/fetch_dependents.py` refreshes that snapshot from GitHub's dependency graph. (#4155)
- An **ONNX, torch.compile and torch.export support** page under Resources (`get-started/export-support`)
  lists every surveyed public operator, grouped by package and section, with its dynamo ONNX export,
  `torch.export` and `torch.compile` outcome, the cause of each failure and a search box with result
  filters. It is rendered at build time from the committed snapshot
  `docs/source/_data/export_support.json`; `docs/export_support/run.py` reruns the survey to refresh
  it. (#4182)
- Documentation pages now provide search-specific titles and descriptions, canonical tutorial links, and a
  canonicalized redirect from the retired highlighted-features page. (#4226)
- Documented bbox/Boxes geometry-operation conventions and added executable pins for transforms,
  NMS, padding, clipping, and area filtering, including known unbatched-input limitations. (#4245)
- Documented camera distortion and calibration conventions (normalized versus pixel inputs, the
  coefficient layout, the `new_K`/`K` roles, the iterative inverses) and added executable pins for
  `kornia.geometry.camera`'s distortion models and `kornia.geometry.calibration`, with the
  tilt-projection, Kannala-Brandt Jacobian and float16 defects tracked in dedicated issues. (#4312)
- Documented camera projection-core conventions (`PinholeCamera` frames, integer pixel centres, the two
  meanings of depth, the `z = 0` policies) and added executable pins for `kornia.geometry.camera`'s
  projection core, with the known scale-rule, aliasing, guard and legacy-API limitations tracked in
  dedicated issues. (#4294)
- Documented depth and stereo conventions (the two meanings of depth, the `(B, 3, H, W)` versus
  `(B, H, W, 3)` layouts, the opposite source/destination naming of `warp_frame_depth` and
  `DepthWarper`, the rectified stereo `Q` matrix) and added executable pins for `kornia.geometry.depth`
  and `StereoCamera`, including dtype promotion, extra-axis broadcasting, subpixel border sampling,
  and singular stereo reprojection. Remaining defects are tracked in dedicated issues. (#4317)
- Documented the `kornia.sensors.camera` conventions (`Vector`-typed inputs, the per-model `params`
  layout, the shared half-pixel `scale` rule, and the pinhole mapping shared with `kornia.geometry.camera`,
  including its camera-axis broadcasting, projection rounding and zero/near-zero depth differences) and
  added executable pins, with the unimplemented models tracked in a dedicated issue; the three non-pinhole
  models now appear on the sensors documentation page. (#4318)
- Documented the shared 2D contract of `kornia.augmentation` — the `(B, C, H, W)` float working layout
  and `keepdim`, `p` / `p_batch` / `same_on_batch`, where random parameters are drawn and how `torch.manual_seed`
  and `params=` replay them, what a module serializes, and what `AugmentationSequential` does with each data key
  (layouts, the inclusive `xyxy_plus` boxes and integer-centre flips, nearest masks, `extra_args`, `inverse`) —
  as Convention blocks on the base classes and containers, with the canonical randomness and serialization
  statements on the Conventions & Pitfalls page and executable pins. The contracts distinguish sampler and
  returned-parameter placement, trainable range parameters, application-time replay exceptions, padding labels,
  information lost by rotated tensor boxes, mask-list gates, mixed mask dtypes, dictionary-key exceptions,
  and the transplantation constructors. Scope corrections and regression tests cover mix-specific contracts,
  inverse support, mask erasing/filtering/precision, nested matrices, serialization and sampler limitations;
  the container, `p_batch`,
  `set_rng_device_and_dtype`, `state_dict` and `B = 0` defects are tracked in dedicated issues. (#4452)
- Dynamo ONNX export (`torch.onnx.export(..., dynamo=True)`) now covers a much larger part of the
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
- Every page in the **Models** section of the docs now opens with the shortest script that runs the
  model and a figure of its output on a real input, with the paper card moved to the bottom. The
  figures are rendered by `docs/generate_model_examples.py` and committed under
  `docs/source/_static/img/models/`, so the docs build needs neither the weights nor the sample
  images. (#4178)
- Implemented the exported Brown-Conrady, Kannala-Brandt K3, and Orthographic sensor camera models,
  including distortion/projection plumbing, intrinsic matrices, and batched project/unproject support. (#4284, #4377)
- KimiVL and SigLIP2 builders load their safetensors checkpoints with kornia's own downloader
  (`kornia.core.download_hf_file`/`download_file_from_url`, which share the retrying, rate-limit-aware
  cache every other checkpoint uses) and a pure-torch reader (`kornia.core.load_safetensors`);
  `huggingface_hub` and `safetensors` are no longer needed. Both were imported without ever being
  declared as dependencies, so `KimiVLBuilder.from_pretrained_hf()` and
  `SigLip2Builder.from_pretrained_hf()` used to raise `ImportError` on a plain `pip install kornia`.
  The checkpoints move to torch's hub cache as a result; see *Breaking changes*. (#4293)
- Linux CPU half-precision CI now uses explicit reproducible profiles, exact phase/exception manifests, complete
  lifecycle verification, and guarded candidate-only baseline regeneration. (#4154)
- New "Camera and world conventions across the ecosystem" page cataloguing the pixel-centre, axis
  and extrinsics conventions of OpenCV, COLMAP, OpenGL, ARKit, ARCore, PyTorch3D and Direct3D with the
  kornia converter for each, plus the baked `align_corners` rows for the depth and undistortion warps on
  the Conventions & Pitfalls page. (#4315)
- Optional-dependency extra `kornia[image]` installs Pillow for PIL-backed image input/output,
  display helpers and `kornia.io.sample`; missing-Pillow errors now name the extra to install, and the unused
  `DinoVisionTransformer.forward_features_list` list path has been removed. (#4464)
- Optional-dependency extras `kornia[onnx]` and `kornia[sd]` declare the third-party packages that
  the ONNX and Stable-Diffusion-dissolving wrappers lazily import, and are documented on the
  installation page. Missing-dependency errors now name the extra to install, and `dev` no longer
  pulls `diffusers` and `transformers`. (#4301)
- Repeatable Oxford affine local-feature benchmarks for SIFT, SIFT-AffNet-HardNet, and
  KeyNet-HardNet, with eager/compiled median/IQR speed, homography corner error, and JSON output;
  historical scale-space SIFT CPU/CUDA batch-runtime comparisons and plotting. (#4254)
- The blocking MPS CI gate now runs on torch 2.14.0 instead of 2.9.1, on the `macos-15` image
  instead of `macos-latest`, and its recorded failure baseline drops from 200 entries to 14.
  torch 2.14.0 adds MPS kernels for `grid_sample`'s 2-D and 3-D backward passes, for
  `padding_mode="border"` and 5-D `mode="nearest"` sampling, and for the eigen/QR/LU/SVD family
  (`eigh`, `svd`, `svdvals`, `qr`, `lu_solve`, `lstsq`, `cholesky_solve`, `matrix_exp`), so most of
  what used to fail on Apple silicon now runs there natively. What is still *pinned* in the
  baseline: `torch.linalg.eigvals`, which has no MPS kernel and keeps the 5-point essential-matrix
  solver off the device, and the `yuv420`/`yuv422` empty-input reshape defect. Separately,
  `tests/geometry/test_ransac.py` is *skipped* rather than pinned — it aborts the process on the
  runners' paravirtualized GPU, and a `SIGABRT` has no exception type to record
  ([#4204](https://github.com/kornia/kornia/issues/4204)); the skip costs 9 tests that pass on real
  Apple hardware, and `--run-mps-process-abort` runs them locally. The underlying limit is that
  `torch.linalg.svd`/`svdvals`/`lstsq` fails to build a Metal pipeline once its input
  holds 8192 elements or more, which used to keep `RANSAC`'s batched minimal solvers off the
  device ([#4201](https://github.com/kornia/kornia/issues/4201)); `_torch_svd_cast` and
  `_torch_linalg_svdvals` now route such inputs through the CPU, so the fundamental and
  homography solvers reach it, while `find_essential` still raises on `torch.linalg.eigvals`,
  which has no MPS kernel. The image stays on `macos-15`: on `macos-latest` (macOS 26) the runner's *virtual*
  GPU cannot compile the Metal 4 cooperative-tensor shaders torch 2.14 emits, which fails 664
  tests — a physical M1 on macOS 26 compiles them fine. kornia still supports torch 2.5.1, so the
  in-tree MPS workarounds stay. (#4202)
- The documentation site was rebuilt on `pydata-sphinx-theme` with a top navigation bar
  (Learn / API / Models plus Ecosystem, About and Support menus), a redesigned landing page, a
  restructured API reference with per-topic subpages, and a long list of fixed doc examples and
  removed dead interactive demos. Old deep links into the split module pages
  (e.g. `augmentation.module.html#kornia.augmentation.RandomAffine`) are forwarded to the subpage
  that now documents the object, so existing links keep resolving. The furo layout stayed available
  behind `KORNIA_DOCS_THEME=furo` until #4304 removed that fallback. (#4155)
- `benchmarks/feature/laf_ops.py` microbenchmarks seven public LAF operations
  (`laf_from_center_scale_ori`, `make_upright`, `ellipse_to_laf`, `laf_to_boundary_points`,
  `laf_is_inside_image`, `extract_patches_simple`, `extract_patches_from_pyramid`) as kornia eager vs
  `torch.compile`, and its CPU and CUDA runs are the first committed non-Apple result set, so the docs
  performance page now shows CUDA numbers. The page labels its throughput column from a new optional
  `units` metadata key (`LAFs/s` here, `items/s` when absent) and identifies a row by every config field
  that varies within a result file rather than by `(op, batch)` alone, which a suite sweeping a second
  axis at a fixed batch would otherwise collapse onto one row. `run_batch_sweep` gained `label_fn`,
  `items_fn` and `units` so such a suite can use the shared sweep; the existing callers are unchanged.
  (#4119)
- `kornia.feature.laf_is_filled(laf)` returns the `(B, N)` mask of the slots a detection filled, the complement
  of a detector's zero-LAF padding. Fixed-shape detectors pad the slots no detection filled with an all-zero LAF,
  and those slots must be dropped before matching -- their descriptors are identical to one another, so they match
  each other at zero distance and a mutual nearest-neighbour test does not reject them. `LocalFeature.forward`'s
  docstring already required callers of a hand-rolled pipeline to do this and gave them the expression to copy;
  they can now call the function instead. It replaces the same expression at four internal sites in
  `kornia/feature`. (#4223)
- `kornia.models.RRDBNet`, a vendored Real-ESRGAN generator; `RRDBNetBuilder` no longer needs `basicsr`. (#4292)

### Bug fixes

- The `kornia.metrics` disparity reductions accumulate in `float32` for `float16` and `bfloat16`
  inputs, so `mean_absolute_disparity_error`, `root_mean_squared_disparity_error` and
  `mean_bad_pixel_error` no longer return `inf` on an image-sized map. A `float16` sum saturates at
  65504, and a `bfloat16` sum lost roughly 0.5% over 100k terms. `root_mean_squared_disparity_error`
  additionally squared in the input dtype, so a 300 px error saturated before any reduction ran;
  it now squares in the accumulator dtype. A mixed-dtype pair now returns the promoted dtype from
  every metric in the module rather than the dtype of `input`, matching what
  `mean_absolute_disparity_error` already did. ([#3931](https://github.com/kornia/kornia/pull/3931))
- `warp_perspective`, `warp_affine`, `homography_warp`, `HomographyWarper` and `warp_image_tps` now
  build their sampling grid and homography normalization under the same `align_corners` convention
  they pass to `grid_sample`. Previously the grid was always corner-aligned, so with
  `align_corners=False` an identity homography did not reproduce its input (max deviation `11.25`
  on a 4x4 `arange` image) and a whole-pixel translation resampled at fractional offsets;
  `warp_image_tps` was affected on its default path. `create_meshgrid`, `normal_transform_pixel`,
  `normalize_homography` and `denormalize_homography` gain an `align_corners` parameter (last,
  default `True`, so existing calls are unchanged). `crop_by_transform_mat` drops the #3650
  correction matrix and the 1-pixel `warp_affine` fallback that worked around the mismatch, so
  1-pixel outputs are exact under both settings. `normal_transform_pixel` also writes the
  `align_corners=False` offset as a single division, so eager and graph capture agree bit for bit at
  every size rather than differing by one `float32` step at 5 of the first 4999.
  Fixes #3904, #3928, #3929, #4411. ([#3945](https://github.com/kornia/kornia/pull/3945))
- Align Kimi-VL positional embeddings, rotary embeddings, and projector ordering with Moonshot. ([#3968](https://github.com/kornia/kornia/pull/3968))
- `build_laplacian_pyramid()` now pads mixed image sizes when needed to keep adjacent pyramid levels aligned, while
  preserving the output shapes of sizes that already worked. ([#3971](https://github.com/kornia/kornia/pull/3971))
- `normal_transform_pixel` and `normal_transform_pixel3d` now raise `ValueError` for an integer
  `dtype` instead of returning a rank-deficient matrix. The scale `2 / (size - 1)` truncated to `0`,
  so `normal_transform_pixel(4, 5, dtype=torch.int64)` returned `[[0, 0, -1], [0, 0, -1], [0, 0, 1]]`
  and mapped every pixel to `(-1, -1)`, with no error or warning. Floating point dtypes, including the
  `float8` formats, and complex dtypes are unaffected. ([#3974](https://github.com/kornia/kornia/pull/3974))
- Homography normalization now builds matrices with float64-native constants for float64 inputs, preserving full precision (#3958).
  `normal_transform_pixel` and `normal_transform_pixel3d` also keep float64 constants exact under TorchScript, where they were previously rounded through float32. ([#3979](https://github.com/kornia/kornia/pull/3979))
- `rotation_matrix_to_quaternion` no longer adds `eps` inside the square root of the dominant component, which inflated every `float64` result: the identity used to return `[1.0000000012499999, 0, 0, 0]` and now returns exactly `[1, 0, 0, 0]`, and the unit-norm error of random rotations drops from about `2.5e-9` to about `7e-16`. `eps` is now only a floor on the radicands of the discarded branches; the selected branch's radicand is at least `1` for any real 3x3 matrix, so it never reaches the floor. `ARKitQTVecs_to_ColmapQTVecs` inherits the fix. `float32`, `float16` and `bfloat16` outputs are unchanged. ([#3981](https://github.com/kornia/kornia/pull/3981))
- `RANSAC`'s local-optimization step now keeps the polished model that `verify()` scores best, instead of always
  taking the first candidate the polisher returns, so the returned model agrees with the returned inliers.
  Every built-in polisher returns a single candidate, so results are unchanged for them; the fix matters for custom
  `polisher_solver`s that return several candidates. ([#3991](https://github.com/kornia/kornia/pull/3991))
- Slicing or masking list-backed `Boxes` now preserves each batch element's padding metadata, so `to_tensor()`
  returns the original number of boxes per element. ([#4194](https://github.com/kornia/kornia/pull/4194))
- `kornia.feature.loftr`'s `CoarseMatching` no longer crashes in training mode. Its gt-padding
  branch called `zip()` with four separate 2-element lists instead of two 4-element lists, so
  every resulting tuple failed to unpack (`ValueError: too many values to unpack`) on every call;
  no existing test exercised this path. Restructured the `zip()` to pair each real tensor with its
  ground-truth counterpart. Once reachable, a second bug surfaced: the padding tensor `mconf_gt`
  was built without `dtype=`, defaulting to float32 regardless of `mconf`'s own dtype (which can be
  float16/bfloat16 under autocast) and silently promoting the concatenated result. Fixed by building
  `mconf_gt` in `mconf.dtype`. ([#4359](https://github.com/kornia/kornia/pull/4359))
- `kornia.enhance.image_histogram2d` no longer collapses distinct bins together on a `float16` or `bfloat16` image. The auto-built bin centers were created at the image dtype, which cannot represent the bin indices exactly: `bfloat16` lost 63 of 256 centers at the default `n_bins`, and `float16` lost 511 of 2048, so those bins came back numerically identical. The centers are now built at `float32` for half-precision images (a `float64` image keeps `float64`), and a floating-point image still gets its own dtype back. Integer images and explicitly passed `centers` behave exactly as before. ([#4360](https://github.com/kornia/kornia/pull/4360))
- `kornia.geometry.subpix.render_gaussian2d` no longer distorts the rendered Gaussian's peak location on large grids for `float16` and `bfloat16` inputs. The pixel-coordinate linspace was built at the input dtype, so distinct coordinates collapsed together once `width`/`height` exceeded what the dtype represents exactly: about 2048 for `float16` and about 256 for `bfloat16`. The coordinate vectors are now built at `float32` for half-precision inputs, and the result is returned in the original dtype. Half-precision outputs of this function change bitwise (in a random-input check against the previous code every grid from 5x7 up differed, and a 1x1 grid did not), so stored regression baselines for half-precision inputs need regenerating. ([#4361](https://github.com/kornia/kornia/pull/4361))
- `PinholeCamera.scale_` now promotes an integer `height` / `width` with a floating-point scale factor to floating point, matching `scale`, instead of raising `RuntimeError`. ([#4371](https://github.com/kornia/kornia/pull/4371))
- `Boxes`, `Boxes3D`, their `from_tensor` helpers and `_transform_boxes` now cast integer
  input to `torch.get_default_dtype()` instead of a hardcoded `float32`. Under a caller that
  sets `torch.set_default_dtype(torch.float64)` — as NPU and other backends commonly do — the
  old `.float()` silently forced every box and transformation matrix down to single precision,
  losing precision end to end. The mask, shape and transform results now follow the configured
  default dtype. ([#4379](https://github.com/kornia/kornia/pull/4379))
- Fixed TorchScript compilation of `unproject_meshgrid`, `depth_to_3d_v2` and
  `warp_frame_depth` by passing a floating-point norm order. ([#4389](https://github.com/kornia/kornia/pull/4389))
- SVD no longer fails on MPS at 8192 elements or more. torch 2.14's MPS `linalg.svd` and
  `linalg.svdvals` fail to build their Metal pipeline state object once an input holds 8192 elements
  or more, whatever the per-matrix shape. At realistic batch sizes `find_fundamental` and `solve_pnp_dlt`
  raised that "Failed to created pipeline state object" exception; `find_homography_dlt(..., solver="svd")`
  instead caught it, emitted a "SVD did not converge" warning, and returned an uninitialized (invalid)
  tensor. Such inputs are now decomposed on the CPU and moved back. `find_essential` still raises on
  MPS at any batch size, because `run_5point` calls `torch.linalg.eigvals`, which has no MPS kernel. ([#4455](https://github.com/kornia/kornia/pull/4455))
- `AutoAugment`, `RandAugment` and `TrivialAugment` now raise a `RuntimeError` from `inverse` when the
  drawn sub-policy applied a non-geometric operation, instead of silently returning a tensor that still
  carries it -- for a draw with no geometry at all, the input unchanged. Only geometric operations are
  invertible, and the sub-policy is redrawn on every forward pass. An operation whose probability gate
  skipped every sample does not count. ([#4456](https://github.com/kornia/kornia/pull/4456))
- Add bfloat16 support to mix augmentations and preserve input half-precision dtypes in MixUp and CutMix. ([#4471](https://github.com/kornia/kornia/pull/4471))
- Built-in 2D augmentations with lazy transformation matrices retain input metadata instead of the
  last image batch, preventing unnecessary tensor lifetime extension and image-sized pickle,
  deepcopy and torch.save payloads. Custom matrix implementations that read pixels retain their
  existing lazy behavior. Cached metadata remains compatible with non-strict torch.export and
  ONNX export after an eager forward. ([#4484](https://github.com/kornia/kornia/pull/4484))
- `RandomCrop` with `padding` or `pad_if_needed` no longer shifts keypoints and bounding boxes by the
  left/top padding when the crop is skipped; they previously moved while the image and mask stayed
  unchanged. (#4473) ([#4485](https://github.com/kornia/kornia/pull/4485))
- `transform_points` returns an empty result for an empty transform batch whose points carry a non-empty
  point axis, such as `(0, 4, 4)` transforms with `(0, 5, 3)` points. It used to raise `ZeroDivisionError`
  expanding the transforms (`0 // 0`), and `PinholeCamera.project` inherited the crash for an empty camera
  batch on `(0, N, 3)` points. (#4466) ([#4487](https://github.com/kornia/kornia/pull/4487))
- `Rotate`, `ShearX`, `ShearY`, `TranslateX` and `TranslateY` now draw negative magnitudes under
  `symmetric_megnitude`, which is their default. The sampled magnitude was multiplied by a boolean
  mask rather than a sign, so these ops never drew a negative magnitude: rows meant to get `-m` got
  `0`, a silent no-op on roughly half of their draws. This reaches the five ops used directly and
  `RandAugment`. `AutoAugment` builds them with `symmetric_megnitude=False` and is unaffected;
  `TrivialAugment` skipped the sign altogether and is covered by #4761 instead (#4441). For a
  seeded pipeline the random stream, the ops picked and the rows that already drew `+m` are unchanged;
  only the rows that drew `0` now draw `-m`. (#4440, #4494) ([#4495](https://github.com/kornia/kornia/pull/4495))
- `AugmentationSequential` dispatches ``mask``, box and ``keypoints`` keys through a mix child's own handlers
  instead of silently returning them unchanged next to a mixed image. Unsupported keys raise
  ``NotImplementedError`` as a direct call does; ``RandomMosaic`` boxes are transformed. (#4493) ([#4505](https://github.com/kornia/kornia/pull/4505))
- `RandomCutMixV2` now documents its `data_keys` constructor argument. ([#4506](https://github.com/kornia/kornia/pull/4506))
- A `resample` entry in `AugmentationSequential(extra_args={DataKey.MASK: {...}})` now reaches the mask sampler on
  2D geometric augmentations that use the base mask path (`RandomAffine`, `RandomRotation`, `RandomPerspective`,
  `Resize`, the crops and the rest). It used to be merged and then overwritten with nearest on the forward path,
  though `inverse` already honoured it. Masks still default to nearest when no `resample` is given, including
  when a user dict leaves it out. (#4419) ([#4509](https://github.com/kornia/kornia/pull/4509))
- `RandomElasticTransform` documents `alpha` in the order the code applies it, `(x, y)`; it said `(y, x)`, the order
  `kernel_size` and `sigma` use, so an anisotropic `alpha` warped the other axis from the one documented. (#4418) ([#4510](https://github.com/kornia/kornia/pull/4510))
- `RandomAutoContrast` documents `clip_output` as having no effect: `normalize_min_max` already maps each channel
  onto `[0, 1]`, so the clamp cannot change a value, including for inputs outside that range. It was described as
  "if true clip output". A constant channel is documented to come back as zeros. (#4436) ([#4511](https://github.com/kornia/kornia/pull/4511))
- `RandomCrop(4)` and `RandomResizedCrop(4)` raise errors that name `size`, the `(height, width)` form it needs and
  the value passed, instead of an `AssertionError` about a `torch.Tensor` shape and a bare
  `TypeError: object of type 'int' has no len()`. The exception types are unchanged, and both classes still reject
  an `int`. (#4417) ([#4512](https://github.com/kornia/kornia/pull/4512))
- `warp_perspective3d` and `homography_warp3d` reproduce an identity warp. `homography_warp3d`
  handed the `(d, x, y)` grid that `create_meshgrid3d` produces straight to `warp_grid3d` and
  `torch.nn.functional.grid_sample`, both of which read the last axis as `(x, y, z)`, so every
  voxel was sampled from the wrong place: an identity homography on a 4x4x4 `arange` volume
  deviated by 45.0 at `align_corners=True` and 55.125 at `False`, and a +1 translation along `x`
  moved slices instead of columns. The grid is reordered at that call site; `create_meshgrid3d`,
  `normalize_pixel_coordinates3d` and `conv_soft_argmax3d` keep their documented `(d, x, y)`
  order and `warp_affine3d`, which builds its grid with `F.affine_grid`, was never affected. The
  `align_corners=False` identity still deviates after this change; that remaining residual is the
  3D normalization convention tracked in #4503. Fixes #4502. ([#4514](https://github.com/kornia/kornia/pull/4514))
- `RandomAffine` and `RandomPerspective` now compile on CUDA with CPU-generated geometry parameters, including numeric and CPU tensor constructor ranges, with or without moving the augmentation. Their parameter placement, returned dtype, and RNG consumption are preserved. This compiler workaround covers these two transforms; other augmentations, including `RandomShear`, `RandomThinPlateSpline`, `RandomErasing`, and `CenterCrop`, may still encounter the upstream CPU-constant transfer defect.

  Augmentations and random generators now use standard module migration: dtype-only `.to()` calls preserve the sampler device, device-only calls preserve its dtype, and invalid integer-dtype requests raise before changing samplers. Generator `.to()`, `.cuda()`, `.cpu()`, `.double()`, and parent-module moves now update registered buffers and samplers together. Plain tensor constructor attributes retain their existing placement precedence. ([#4517](https://github.com/kornia/kornia/pull/4517))
- The unimplemented tuple-bound `Boxes.clamp`, `Boxes.trim`, and `Boxes.translate(method="fast")` paths now raise
  `NotImplementedError` with actionable messages, including the supported tensor bounds and `warp` alternatives where applicable. ([#4518](https://github.com/kornia/kornia/pull/4518))
- Extend the CUDA `torch.compile` constant-transfer workaround to shear, thin-plate spline, erasing, crop, resize, 3D affine/perspective, and illumination/mix parameter generators. Eager execution, `torch.compile`, and `make_fx` build these constants with the same scalar factories, without a host-device copy. Crop operations retain their existing graph breaks.

  Fix `RandomShear` and `RandomAffine` sampler migration so scalar and tuple shear ranges honor the requested device and dtype; moving `RandomAffine` with a scalar or two-element `shear` to CUDA used to raise a device-mismatch error. Fix mixed-device sign generation in both linear-illumination generators after moving their samplers. `CenterCrop3D` parameters for an empty batch are now `long` tensors on the requested device, matching non-empty batches, instead of CPU `float32` tensors. ([#4523](https://github.com/kornia/kornia/pull/4523))
- `RandomPerspective3D`'s docstring example printed the output of the 3D sampling-grid defect fixed
  in #4514, so its values were a permutation of the correct ones. The example now shows what the
  class returns. Only the docstring changes; no behaviour does. ([#4539](https://github.com/kornia/kornia/pull/4539))
- `Resize(antialias=True)` no longer antialiases masks, so their label set is preserved; pass `antialias=True` in `extra_args[DataKey.MASK]` (or to `transform_masks`) to filter soft masks explicitly. ([#4540](https://github.com/kornia/kornia/pull/4540))
- `kornia.morphology.dilation`/`erosion` (and `opening`, `closing`, `gradient`, `top_hat`, `bottom_hat`) no longer crash with `engine="convolution"` when the input tensor's dtype differs from the kernel's dtype; the result now preserves the input tensor's dtype. ([#4544](https://github.com/kornia/kornia/pull/4544))
- `KMeans.fit` no longer synchronises with the device once per cluster per iteration to update centroids; the update is now vectorised. This mainly benefits GPU backends where the per-cluster host syncs were expensive (e.g. MPS); on CPU the win is smaller and shrinks further at larger problem sizes, where the O(N·K·D) distance computation dominates. A fit that hits an empty cluster may now produce different (but equally valid) centers for the same seed, since the random reseed draws differ from before. Two more observable side effects of the old in-place update are also gone: `fit` no longer mutates a `cluster_centers` tensor passed in at construction, and calling `fit` a second time no longer warm-starts from the first call's result - neither was documented behavior, but code relying on either will now see a difference. ([#4545](https://github.com/kornia/kornia/pull/4545))
- `RandomLinearIllumination` and `RandomLinearCornerIllumination` build their `gradient` parameter on
  the device and dtype their samplers were moved to, matching `RandomGaussianIllumination`. Moving
  either augmentation to CUDA or MPS, or calling `set_rng_device_and_dtype` on its generator as the
  class docstring describes, used to return a CPU tensor of the default dtype, so a `float64` pipeline
  silently got `float32` parameters and every call copied the gradient across devices. (#4536) ([#4546](https://github.com/kornia/kornia/pull/4546))
- `kornia.losses.total_variation` reduces over one flattened dimension instead of `dim=(-2, -1)`, which returns identical values and avoids a slow reduction path on MPS. ([#4547](https://github.com/kornia/kornia/pull/4547))
- `kornia.color.rgb_to_yuv420` subsamples chroma with `avg_pool2d` instead of a mean over two unfolded window dimensions: same values, several times faster on CPU and MPS. ([#4548](https://github.com/kornia/kornia/pull/4548))
- Calling `transform_boxes` directly on a 2D intensity augmentation returns the boxes unchanged instead of raising `NotImplementedError`: `IntensityAugmentationBase2D` now defines its box pass-through handlers under the names the dispatcher calls. ([#4549](https://github.com/kornia/kornia/pull/4549))
- `RandomMosaic` places bounding boxes on the tile that holds their image on non-square inputs: the x and y tile offsets of the box composition were swapped, which only agreed with the composed image when height equals width. ([#4550](https://github.com/kornia/kornia/pull/4550))
- SOLD2's line matcher now samples line descriptors at the positions the reference implementation
  uses. `keypoints_to_grid` normalized keypoints with the corner-aligned mapping and fed them to
  `grid_sample(align_corners=False)`, so every sample drifted by `p / (size - 1)` image pixels toward
  the far edge relative to the grid the pretrained weights were trained with, up to one image pixel
  and one eighth of a descriptor pixel on a 480x640 image. It now uses the reference formula
  `keypoints * 2 / img_size - 1`, under which image pixel `p` reads descriptor coordinate
  `p / grid_size - 0.5`. Match decisions can change anywhere except at the top-left origin; the shift
  grows from zero there to one image pixel at the far edge. Fixes #4554. ([#4558](https://github.com/kornia/kornia/pull/4558))
- `rgb_to_hsv` and hue-based color augmentation now keep black pixels and their gradients finite in float16. ([#4561](https://github.com/kornia/kornia/pull/4561))
- Fix float16 NaNs in thin-plate-spline transforms by handling zero kernel distances safely. ([#4565](https://github.com/kornia/kornia/pull/4565))
- `RandomRain` now draws `number_of_drops`, `drop_height` and `drop_width` uniformly over their closed integer
  ranges. The upper bound used to be practically unreachable (`drop_height=(5, 20)` gave 5 to 19) and a signed
  `drop_width` drew `0` twice as often as any other value (#4567). Seeded draws therefore differ from earlier
  releases, and because the upper bounds are drawn now, an image that only cleared the old effective maximum can
  raise the documented drop-size check: on the defaults, a 20-row or 5-column image raises on some seeds where it
  never did before. A range that is reversed, fractional or non-finite is rejected at construction instead of
  reaching the sampler. ([#4568](https://github.com/kornia/kornia/pull/4568))
- `RandomPlanckianJitter` now preserves the input dtype. The illuminant table was moved to the input's
  device but left at `float32`, so multiplying the red and blue channels by it promoted a `float16` or
  `bfloat16` input to a `float32` output; the table now follows the input's dtype as well. `float32` and
  `float64` outputs are bit-identical. Fixes #4574. ([#4578](https://github.com/kornia/kornia/pull/4578))
- `normalize` and `normalize_min_max` now accept non-contiguous images, including through the `Normalize` and `RandomAutoContrast` augmentations. ([#4580](https://github.com/kornia/kornia/pull/4580))
- Fixes `augmentation.Denormalize` so integer `mean` and `std` values are accepted alongside floating-point, list, tuple, and tensor statistics. ([#4581](https://github.com/kornia/kornia/pull/4581))
- `RandomClahe` now applies each sampled clip limit to its corresponding image in a batch. ([#4582](https://github.com/kornia/kornia/pull/4582))
- Use safe denominators in `rgb_to_hls` for achromatic inputs and out-of-range colors with lightness zero or one.
  For achromatic inputs in every supported float dtype, hue-gradient magnitudes now scale as `pi / 3`
  instead of `pi / (3 * eps)`, reducing them from about `1e8` to about `1` with the default `eps`;
  in `float16` they were `nan`/`inf` and are now finite. ([#4583](https://github.com/kornia/kornia/pull/4583))
- `RandomPlasmaBrightness`, `RandomPlasmaContrast`, and `RandomPlasmaShadow` now share one sampled map
  when `same_on_batch=True` (#4570). Seeded outputs and the subsequent RNG stream therefore differ
  from earlier releases for batches with multiple samples; `same_on_batch=False` is unchanged. ([#4584](https://github.com/kornia/kornia/pull/4584))
- `kornia.enhance.equalize_clahe` and `kornia.augmentation.RandomClahe` now raise a `RuntimeError`
  naming the `[0, 1]` input range for values the 256-bin tile lookup cannot index, instead of a raw
  `index ... is out of bounds for dimension 5 with size 256` from the gather — the same named-range
  treatment `equalize` got in #4489. The check uses `torch._assert_async`, so it adds no device sync
  and `torch.compile` fullgraph still works; inputs the lookup can index (including values a hair
  above 1) are unchanged. ([#4585](https://github.com/kornia/kornia/pull/4585))
- RectangleEraseGenerator now keeps position sampling in safe floating-point precision and preserves nonnegative erase positions for half and bfloat16 parameters. (#4553) ([#4587](https://github.com/kornia/kornia/pull/4587))
- RandomCrop now uses the effective pre-crop canvas for oversized requests. With `RandomCrop((10, 10), padding=1)` on a 3-by-3 input, the scale changes from `10/3` to `10/5`; the no-padding behavior tracked in #4414 is unchanged. ([#4588](https://github.com/kornia/kornia/pull/4588))
- `RandomBoxBlur` accepts the same `border_type` spellings as `RandomGaussianBlur` and `RandomMotionBlur`: a `BorderType`, its integer value or an upper-case name no longer raises, and an unknown name is rejected when the augmentation is constructed. `RandomBoxBlur.flags["border_type"]` now holds a `BorderType` rather than the string that was passed in. ([#4591](https://github.com/kornia/kornia/pull/4591))
- `kornia.filters.kernels.gaussian` no longer returns NaN when every sample underflows, which a small `sigma` (or `sigma = 0`) on a short window caused. It returns the unit-impulse limit instead, so `RandomGaussianIllumination` produces a finite image for every `sigma` its constructor accepts. Kernels with an odd window are bit-identical to before; an even window can differ in the last bit (half a float32 epsilon at worst over window sizes 1 to 31), because the nearest sample now sits a quarter-pixel away and that subtraction rounds. The impulse limit is differentiable: the gradient with respect to a tensor `sigma` is 0 there, the value of the continuous `sigma -> 0` limit, rather than NaN. An empty window still returns an empty kernel. ([#4592](https://github.com/kornia/kornia/pull/4592))
- `draw_line` now always draws both endpoints: it could stop one pixel short of `p2`, or pass a row or column off a pixel centre the line runs exactly through, because the minor coordinate was stepped in floating point. It is now placed at `ceil(t * minor / major)` in integers, one pixel per step along the major axis. `draw_line` also builds every line's pixels on the image's device instead of in Python loops over 0-d tensors, accepts an empty `(0, 2)` batch, and rejects a `p1` and `p2` of different shapes instead of broadcasting a `(2,)` point against a `(B, 2)` batch. ([#4596](https://github.com/kornia/kornia/pull/4596))
- `MKDDescriptor`'s `ExplicitSpacialEncoding` computes the spatial encoding with one matrix multiply instead of building a `(B, in_dims * d_emb, H, W)` product and summing it over the spatial dimensions. Descriptor values are unchanged and existing state dicts still load. `ExplicitSpacialEncoding` now raises `ValueError` for an input whose channel count differs from `in_dims`, as documented; its shape check could never fire. ([#4598](https://github.com/kornia/kornia/pull/4598))
- Fixed `RandomDissolving` and `StableDiffusionDissolving` docstring figure tables to render images instead of literal text, and updated the SD XL column to show a pending asset note instead of a mismatched SD 2.1 image. ([#4603](https://github.com/kornia/kornia/pull/4603))
- `RandomChannelDropout` accepts an `int` `fill_value` such as `0` or `1`, as `RandomErasing` does for `value`, instead of raising `TypeCheckError`. The value is stored as a float buffer, so the output is identical to passing the float. ([#4607](https://github.com/kornia/kornia/pull/4607))
- `kornia.enhance.solarize` accepts `additions` of exactly `-0.5` and `0.5`, the closed range `RandomSolarize` already admits at construction. `RandomSolarize(additions=(-0.5, 0.5))` no longer raises `The addition must be in the open range (-0.5, 0.5)` on the forward pass when it draws an endpoint, or on every call for an endpoint-only range. ([#4608](https://github.com/kornia/kornia/pull/4608))
- `RandomMotionBlur` now draws every odd size in a `kernel_size` range with equal probability, the upper bound included. Previously the upper bound was practically never drawn (only when a float32 draw rounded onto it), so `kernel_size=(3, 5)` practically always blurred with `3` and `(3, 7)` drew practically only `3` and `5`. An even upper bound caps at the largest odd size below it, and a range holding no odd size still rounds up, so `(4, 4)` draws `5` as before. Because the larger sizes are now drawn, `kernel_size=(3, 5)` with `border_type="reflect"` on a `2 x 2` image raises for roughly half the draws, where the constant `3` always fit. A two-element list such as `kernel_size=[3, 9]` is now accepted as a range, like a tuple, instead of raising `TypeError`. ([#4610](https://github.com/kornia/kornia/pull/4610))
- `equalize_clahe`, and so `RandomClahe`, builds every tile histogram with one `scatter_add_` instead of one `torch.histc` call per tile, which is 192 kernel launches for a 256x256 RGB image on the default 8x8 grid. The histograms and the output are bit-identical to before on CPU. On MPS the tile histogram now matches the CPU one for inputs marginally outside `[0, 1]` (for example `1.0000001` or `-1e-7`), which `torch.histc` on MPS counted and the CPU kernel skips. ([#4611](https://github.com/kornia/kornia/pull/4611))
- `RandomRain` now places drops uniformly over every start position that keeps the drop inside the image, so the last row and column can be painted. ([#4612](https://github.com/kornia/kornia/pull/4612))
- `axis_angle_to_rotation_matrix` returns finite gradients near the identity in `float16`; the discarded Rodrigues branch no longer differentiates `sqrt(0)` behind a `1e-12` floor that underflows to zero. ([#4613](https://github.com/kornia/kornia/pull/4613))
- `LoFTR` fine matching returns finite gradients through `expec_f` in `float16` on torch older than 2.14. The `1e-10` floor under `sqrt` underflows to zero in `float16`, so a peaked heatmap -- what a confident match produces, not a degenerate corner -- sent `sqrt'(0) = inf` back into the features; since `std` is summed into the loss, one such match is enough to make the whole gradient `nan`. ([#4614](https://github.com/kornia/kornia/pull/4614))
- `in_range`, `normalize_min_max`, `posterize`, `sharpness`, `equalize`,
  `equalize_clahe`, `jpeg_codec_differentiable` and `equalize3d` now accept
  non-contiguous inputs of more than 4 (resp. 5) dimensions instead of raising a raw
  `view size is not compatible` error. ([#4615](https://github.com/kornia/kornia/pull/4615))
- Fixed the mixed-precision (`autocast`) guard in `kornia.feature.lightglue` to resolve the accelerator type from the input tensor instead of a hardcoded `"cuda"`. This makes `LightGlue`/`LighterGlue` usable under autocast on any accelerator (Ascend NPU, XPU, MPS) and on CPU-only builds where `torch.accelerator.current_accelerator()` returns `None`. (#4624) ([#4624](https://github.com/kornia/kornia/pull/4624))
- Fixed non-finite gradients in `rotation_matrix_to_quaternion` and `rotation_matrix_to_axis_angle` near singular points (identity and 180° rotations) across `float16`, `bfloat16`, and `float32` by applying safe square-root gradient substitution matching #4613. Fixed `normalize_quaternion` gradient NaN leakage for exact zero-norm inputs and `eps=0.0` (note: subnormal-but-nonzero norms in `float16` remain out-of-scope pre-existing behavior). ([#4627](https://github.com/kornia/kornia/pull/4627))
- `equalize_clahe` and `RandomClahe` accept a `grid_size` whose two entries differ. Such a grid used to raise
  `IndexError: shape mismatch: indexing tensors could not be broadcast together` on every image (#2531). Output
  for square grids is unchanged. ([#4628](https://github.com/kornia/kornia/pull/4628))
- Fix `equalize_clahe` transpose equivariance in the slow differentiable path when histogram clipping is enabled. The fast histogram path preserves its existing redistribution behavior, including for marginally out-of-range input values. ([#4637](https://github.com/kornia/kornia/pull/4637))
- Setup documentation (`CONTRIBUTING.md`, `README.md`, `README_zh-CN.md`, `AGENTS.md`, `docs/export_support/README.md`) now qualifies the ambiguous Pixi tasks with `-e default`, so `pixi run install`, `pixi run install-docs` and `pixi run build-docs` work as documented instead of stopping with a "task is ambiguous" error or an environment picker. ([#4639](https://github.com/kornia/kornia/pull/4639))
- Speed up large float32/float64 CPU Gaussian blurs and scale pyramids on builds without oneDNN while preserving tensor sigma gradients and existing autocast/export behavior. ([#4641](https://github.com/kornia/kornia/pull/4641))
- `equalize_clahe(..., slow_and_differentiable=True)` no longer raises `IndexError` for `grid_size=(1, 1)`, which is global histogram equalization with the clip limit applied. Grids with more than one tile are unchanged. ([#4642](https://github.com/kornia/kornia/pull/4642))
- `RandomBoxBlur`, `RandomGaussianBlur` and `RandomSharpness` now raise a `ValueError` naming the class, the kernel, the border type and the input shape when the image is too small for the filter, instead of letting a raw torch padding error out. The sizes they accept are unchanged: the minimum is derived from the wider of `filter2d`'s two pads, which for an even kernel extent is the rear one, so `RandomBoxBlur((4, 4), border_type="reflect")` reports the 3 pixels it needs rather than 2. ([#4645](https://github.com/kornia/kornia/pull/4645))
- Speed up median blur inference for 3×3 and 5×5 kernels using selection networks on CPU, and on CUDA for 3×3 kernels or under `torch.compile`. Speed up CPU float32/float64 Laplacian filtering using separable neighborhood sums (with oneDNN, below 2^22 input elements), also on CUDA under `torch.compile`. Preserve median gradient tie behavior with the original autograd path. Replace dense and separable box convolutions with average pooling for floating-point inputs outside autocast, except where oneDNN's CPU convolution is faster: eager inputs below 2^22 elements, larger windows, and compiled windows wider than 5 (separable) or 3 (dense). Construct fixed first-order Sobel kernels directly without transposing, stacking, or normalizing them. Benchmark timing now honors the configured PyTorch thread count, and the filters flagship warms the CPU thread pool with sustained load so hybrid CPUs are measured on their performance cores. Use the large-image Gaussian slice shortcut in eager mode on CPUs with oneDNN, under `torch.compile` on CPUs without oneDNN, and under `torch.compile` on CUDA. Avoid padded-image copies for odd constant-border box windows, and redundant identity passes for one-dimensional box windows. Reuse Gaussian CPU accumulation buffers during eager execution (also accelerating unsharp masking), keep bilateral patch axes as views to avoid materialized patch copies, fuse guided-filter multiply/add operations, and vectorize 2D motion-kernel construction. Ordinary eager CPU box/Laplacian filtering uses a range check and convolution fallback for extreme values, preserving the optimized path for practical image ranges; captured graphs and function transforms keep their native arithmetic. ([#4647](https://github.com/kornia/kornia/pull/4647))
- Augmentation classes no longer exhaust a shared `torch.compile` cache. Compiled
  `RandomResizedCrop` and `RandomCrop` slice modes keep sampled crop coordinates in
  tensors, avoiding recompilation as crop boxes change while preserving interpolation
  and input gradients. Nearest-neighbor upsampling accumulates half-precision
  input gradients in float32 to avoid saturation. Compiled `RandomClahe` also keeps per-image clip limits in
  tensors so varying sampled limits reuse the graph.

  `RandomClahe` batches clip limits in eager execution as well. Compiled crop
  interpolation matches eager to floating-point tolerance rather than bitwise.
  Avoiding recompilation does not guarantee faster steady-state execution: compiled
  CPU `RandomCrop` can remain slower than eager or the previous post-fallback path. ([#4659](https://github.com/kornia/kornia/pull/4659))
- RandomMixUpV2 and RandomCutMixV2 preserve class-label precision when the input image uses float16 or bfloat16. ([#4661](https://github.com/kornia/kornia/pull/4661))
- Fix `RandomMotionBlur3D` ranged kernel sizes for batched inputs. ([#4662](https://github.com/kornia/kornia/pull/4662))
- `equalize`, `equalize_clahe`, `RandomEqualize` and `RandomClahe` now raise the same `RuntimeError` naming the
  `[0, 1]` input range on MPS as on the CPU, instead of torch's raw `gather: index ... is out of bounds` error.
  On MPS the check is the same asynchronous `torch._assert_async` from torch 2.13; older releases have no
  MPS kernel for it, so there the condition is read on the host at one device sync per call and is skipped
  under `torch.compile`. ([#4663](https://github.com/kornia/kornia/pull/4663))
- `AugmentationSequential` returns each mask in the dtype of its own argument. It used to cast every mask output to the
  dtype of the last mask argument, so an integer semantic mask followed by a boolean one came back boolean and lost its
  labels. An empty batch with a mask no longer raises `IndexError`, and a mask placed before the image in dictionary
  insertion order is now converted with that call's image dtype instead of the previous call's, so a float64 mask keeps
  its precision. The same holds under `torch.export`, which used to convert masks through `float32` or a dtype left
  behind by an earlier eager call and bake the rounding into the exported graph. ([#4665](https://github.com/kornia/kornia/pull/4665))
- `RandomCrop3D` now preserves the input when the transformation is skipped, including when `padding` or `pad_if_needed` is enabled. ([#4667](https://github.com/kornia/kornia/pull/4667))
- `solve_quartic` no longer returns values that do not satisfy the quartic it was given. When the
  resolvent cubic has a near-double root, `solve_cubic` can lose it, and Ferrari then factored a
  different polynomial and reported its roots: `(x^2 + 2x + 5)(x^2 + 2x + 5.01)`, which has no real
  roots, came back as `[1.0012, -3.0012, 1.0012, -3.0012]` with a residual of 64. Each candidate is now
  Newton-polished and checked against the quartic with a `sqrt(eps) / 4` tolerance for its dtype. A
  zero that stood for a complex pair is polished too, but kept only where Newton converged at a simple
  root, and a root reached by two candidates is reported once, from Ferrari's own candidate. Over
  50k-row families built from known roots, rows returning a non-root on the near-square family fall
  from 23,488 to 0 in float32 and from 10,727 to 0 in float64, no family gains a row with a value off
  its root, and the small root of a mixed-scale quartic that `main` misses is recovered (rows missing
  a root fall from 18,495 to 1,851 in float32). The residual is measured against the size of the
  quartic's terms with a floor of 1, as elsewhere in `solve_quartic`, so the check does not reach a
  quartic whose roots are all small: the near-square family scaled to roots of about 0.04 still
  returns non-roots in float32, and scaled to about 0.004 in float64. Affects float64 as well as
  float32. Fixes #4474. ([#4669](https://github.com/kornia/kornia/pull/4669))
- `RandomMotionBlur3D` keeps a tuple `kernel_size` inside the range it was given. An even upper bound used to
  admit the odd size above it -- the `+ 1` was taken off `kernel_size[1] // 2` rather than off the largest odd
  size in the range -- so `(3, 4)` drew `5` and `(3, 20)` drew `21`. The upper bound is now the largest odd size
  not above `kernel_size[1]`. A `float32` draw landing on the sampler's open upper end is clamped back onto the
  range, where `(3, 5)` could otherwise draw `7`. A reversed range such as `(20, 3)` now raises a `ValueError` at
  construction; it used to be accepted and drew a constant `9`, outside both of its own bounds. A range holding
  no odd size still rounds up out of the range, so `(4, 4)` draws `5` as before, matching the 2D generator. (#4674) ([#4674](https://github.com/kornia/kornia/pull/4674))
- `RandomRain` now rasterises every drop of every sample in a single indexed write instead of one
  `index_put_` per step of each drop per sample, removing the per-sample host-to-device copies and
  `.item()` syncs as well. Output is unchanged: verified bit-identical (`torch.equal`) against the
  previous implementation over 12,482 configurations across four device/dtype combinations (cpu
  float32 and float64, mps float32 and float16). On MPS at batch 32
  (`128 x 128`) this is 45.0 ms to 5.1 ms, and at batch 8 (`256 x 256`) 12.0 ms to 3.6 ms; CPU is
  faster at every shape measured. Fixes #4530. ([#4675](https://github.com/kornia/kornia/pull/4675))
- Mix augmentations now reject unsupported data keys even when no samples are selected.
  RandomJigsaw validates that image dimensions are divisible by its grid regardless of the probability gate. ([#4676](https://github.com/kornia/kornia/pull/4676))
- Fix RandAugment posterize and translation magnitudes. Posterize now uses the 4–8 bit range, preventing low magnitudes from producing black images, and translation magnitudes are converted from image-relative fractions to pixel values before being applied. ([#4677](https://github.com/kornia/kornia/pull/4677))
- `OperationBase` can now be deep-copied after augmentation operations have been used without retaining stale probability sampler state. ([#4678](https://github.com/kornia/kornia/pull/4678))
- `RandomMosaic` now respects `output_size` when transforming bounding boxes and no longer crashes with
  `cropping_mode="resample"` when `output_size=None`. With an explicit `output_size`, both cropping modes now take an
  unscaled `output_size` window of the composed canvas, so the boxes stay on their tiles; before, `"slice"` rescaled an
  input-sized crop whenever every sample drew the same corner and zero-padded it otherwise. When the gate selects only
  part of the batch, an unselected sample's boxes are now returned unchanged; before, they were clipped to the input
  extent and a box below `min_bbox_size` was replaced by `[0, 0, 1, 1]`. For a tensor box input the unselected row is
  still padded to the mosaic box count, now with all-zero (zero-area) rows instead of the `[0, 0, 1, 1]` placeholder
  that looked like a real box (inside `AugmentationSequential`, `"bbox_xyxy"` padding was already all-zero). A list box
  input still comes back as a list with one tensor per sample, and an unselected sample's tensor now holds only its own
  boxes instead of being padded with placeholder rows. Fixes #4652. ([#4679](https://github.com/kornia/kornia/pull/4679))
- `PatchMix` rejects a `patch_size` larger than the input instead of drawing negative corners.
  The corner range was `height - patch_size` with nothing checking that the patch fits, so a
  larger patch made it negative and the negative slice that followed copied an arbitrary
  rectangle: the default `patch_size=16` on an 8x10 input changed a 7x5 region, and
  `patch_size=9` changed a non-square 8x9 one, neither raising. `same_on_batch=True` also
  returned its input unchanged, because the pairing is an argsort of drawn values and sharing
  that draw made every value equal, so the permutation was the identity and each image was
  patched with itself. The pairing draw is no longer shared; the patch location still is. ([#4680](https://github.com/kornia/kornia/pull/4680))
- `kornia.enhance.normalize` and the `Normalize` augmentation now preserve empty batches instead of raising a reshape error for `B=0`. Existing `mean` and `std` validation remains unchanged. ([#4681](https://github.com/kornia/kornia/pull/4681))
- `kornia.geometry.cart2pol` now returns the exact Euclidean radius without perturbing the forward value, while keeping the gradient finite at the origin. ([#4682](https://github.com/kornia/kornia/pull/4682))
- Fix double application of probability `p` in `RandomMixUpV2` and `RandomCutMixV2`. ([#4685](https://github.com/kornia/kornia/pull/4685))
- `RandomMotionBlur` now draws a ranged kernel size once per call and repeats it across `ksize_factor`, so the stored sizes match the shared size applied to the batch. Previously it drew a size per sample and used only one randomly selected entry. Angle and direction still follow `same_on_batch`. Fixed integer kernel sizes and replay of previously saved parameters are unchanged; seeded ranged calls with `same_on_batch=False` can produce different sizes and advance the random generator differently. ([#4687](https://github.com/kornia/kornia/pull/4687))
- `RandomCrop3D` no longer accepts a crop exactly one voxel larger than the (padded) input along an axis. It used to pass the size guard, crop past the edge and return a volume with an empty slab; it now raises the same `ValueError` that a larger crop already did. `RandomCrop` and `RandomCrop3D` also honour a `fill` sequence holding one value per channel, which both have always documented but which raised `TypeError: pad(): argument 'value' must be float, not tuple`; a sequence must match the channel count and requires `padding_mode="constant"`. In `AugmentationSequential`, a per-channel image fill defaults mask padding to zero unless `extra_args[DataKey.MASK]["fill"]` overrides it, so masks with a different channel count remain supported. ([#4691](https://github.com/kornia/kornia/pull/4691))
- `RandomTransplantation` no longer misaligns donor labels when one donor's labels are all excluded. The label list was built only for donors that had an eligible label and then zipped positionally against the donors, so a single fully excluded donor shifted every later label onto the wrong donor and dropped the last acceptor — at a batch size of two the whole call became a silent no-op, and at three an acceptor could receive the region chosen for a different donor. An acceptor whose donor has no eligible label is now dropped from `acceptor_indices` and `donor_indices` and its `batch_prob` entry is cleared, so the remaining labels stay aligned with their own donors for every mask dtype, and an empty `selected_labels` follows the mask's dtype and device instead of defaulting to float32 on CPU. Labels are no longer drawn, and the RNG no longer consumed, when the caller supplies a `selection`. The length-check message no longer reads "than images torch.where this augmentation should be applied". ([#4695](https://github.com/kornia/kornia/pull/4695))
- `AugmentationSequential` now rejects 4D image batches when used with a 3D augmentation instead of silently treating the batch as a single 3D sample. ([#4697](https://github.com/kornia/kornia/pull/4697))
- Fixed `AugmentationSequential.inverse()` silently ignoring unsupported mix augmentations. ([#4698](https://github.com/kornia/kornia/pull/4698))
- `RandomJigsaw(ensure_perm=True)` now rejects the permutation that reproduces the image. It used to reject
  `[0, ..., N - 1]`, which is the no-op only for a single-row or single-column grid: on a `2 x 2` grid it is a
  transpose of the patch grid, while the no-op `[0, 2, 1, 3]` was still drawn, so about 1 call in 23 returned its
  input unchanged. The rejected permutation is now `arange(N).view(rows, columns).T.flatten()`. A `1 x 1` grid with
  `ensure_perm=True`, which used to hang in the rejection loop, now raises `ValueError` in the constructor. Fixes #4703. ([#4708](https://github.com/kornia/kornia/pull/4708))
- `ColorJiggle` and `adjust_saturation` now avoid a redundant RGB-to-HSV round trip in float32 and float64, improving eager and compiled performance. At tied RGB extrema, the direct formula uses PyTorch's symmetric max/min subgradient instead of the arbitrary first-channel subgradient from the HSV branch. ([#4717](https://github.com/kornia/kornia/pull/4717))
- `adjust_brightness`, `adjust_contrast`, `adjust_saturation`, `adjust_hue` and their variants now raise `ValueError` for a `factor` with more dimensions than the image, instead of hanging forever (#4718). ([#4720](https://github.com/kornia/kornia/pull/4720))
- `RandomCutMixV2` now draws the cut placement per row and mix when `same_on_batch=False`; one uniform draw per axis used to place every cut of the batch, so all cuts sat at the same relative position. The random stream for a fixed seed changes when `B * num_mix > 1`. ([#4721](https://github.com/kornia/kornia/pull/4721))
- `ColorJiggle(order=...)` no longer turns off `torch.distributions` argument validation for the rest of the process. Its eager `torch.cond` dispatch entered Dynamo, whose one-time setup disables that validation globally; the caller's setting is now restored. A fixed `order` also no longer raises `torch.cond requires dynamo support` where Dynamo is unavailable (torch 2.5.1 on Python 3.13); it uses the Python dispatch there. ([#4725](https://github.com/kornia/kornia/pull/4725))
- `RandomMixUpV2` and `RandomCutMixV2` now replay a `forward_parameters()` dictionary with the `"class"` key; the image dtype that the label handlers need is taken from the input, instead of raising `KeyError: 'dtype'`. A replayed dictionary with an unsupported image dtype still raises `TypeError`. ([#4726](https://github.com/kornia/kornia/pull/4726))
- `RandomAffine3D` and `AffineGenerator3D` now draw a two-value `scale=(a, b)` once per sample and apply it to all three axes, as the docstring says and as the 2D `RandomAffine` does; each axis used to be drawn independently. The three-pair form `((a, b), (c, d), (e, f))` is unchanged. The random stream for a fixed seed changes for the two-value form: the scale consumes one draw instead of three, so the translations and shears drawn after it move. ([#4731](https://github.com/kornia/kornia/pull/4731))
- `CenterCrop3D` and `RandomCrop3D` now accept a crop with a size-1 axis. The crop matrix is built as the translation it is, instead of being solved from the box vertices, which raised `_LinAlgError` because those vertices are coplanar. ([#4732](https://github.com/kornia/kornia/pull/4732))
- `kornia.geometry` no longer re-exports `sys`, `struct`, `array`, `os`, `BinaryIO`, `Dict`, `List`, `NamedTuple` and `is_compiling` from its `pointcloud` and `grid` submodules, which now declare `__all__`. ([#4738](https://github.com/kornia/kornia/pull/4738))
- `AugmentationSequential.inverse` now inverts a list-valued `mask` element by element, reusing each entry's forward gate, instead of raising `AttributeError: 'list' object has no attribute 'dtype'`. ([#4739](https://github.com/kornia/kornia/pull/4739))
- `RandomTransplantation` and `RandomTransplantation3D` accept a `(B, 1, *spatial)` mask next to an image of the same rank and read it as the `(B, *spatial)` mask, so `RandomTransplantation` no longer raises inside `AugmentationSequential` when another augmentation runs before it. ([#4740](https://github.com/kornia/kornia/pull/4740))
- `crop_by_indices`, `RandomCrop` and `RandomResizedCrop` return an empty batch for an empty batch instead of raising `IndexError: list index out of range`. ([#4741](https://github.com/kornia/kornia/pull/4741))
- Corrected the `kornia.morphology` docstrings: `origin` defaults to `[k_h // 2, k_w // 2]` rather than
  "the center ... (rounding towards zero)", the `Returns` section of `bottom_hat` names the bottom hat rather than
  the top hat, and the kernel shape is `(k_h, k_w)` over the image's `(H, W)` axes rather than `(k_x, k_y)`. ([#4743](https://github.com/kornia/kornia/pull/4743))
- `CenterCrop`, `RandomCrop`, `center_crop` and `CenterCrop2D` now accept a crop with a size-1 axis. The crop matrix is built as the translation it is, instead of being solved from the box vertices, which returned a matrix of `NaN` because those vertices are collinear. For every other crop the matrix is now the exact translation rather than the solver's approximation of it, so `transform_matrix`, transformed boxes and keypoints, and resampled pixels can differ from before in the last bits in float32, and by more in bfloat16, where the solve was inexact for coordinates above 256. ([#4753](https://github.com/kornia/kornia/pull/4753))
- `draw_point2d` accepts a single `[x, y]` vector and an empty `(0, 2)` point set instead of raising `TypeError` / `ValueError`. ([#4754](https://github.com/kornia/kornia/pull/4754))
- `RandomCutMixV2` now draws each cut start uniformly over every legal position, `0` to `side - cut_side` on each
  axis. Previously the draw stopped short of the far edge: with the default float32 parameters a cut smaller than the
  image never reached its last row or column, and a full-image cut (`cut_size=(0.0, 0.0)`) practically always started
  one pixel before the image, so the last row and column kept their original pixels while lambda reported a full-image
  cut. The random stream for a fixed seed changes: cut sizes, lambdas and `mix_pairs` are unchanged, but the cut
  positions differ from earlier releases. ([#4757](https://github.com/kornia/kornia/pull/4757))
- `dilation`, `erosion`, `opening`, `closing`, `gradient`, `top_hat` and `bottom_hat` now ignore `border_value` under `border_type` `reflect`, `replicate` and `circular`, as documented, instead of raising `RuntimeError: Padding mode "reflect" doesn't take in value argument` for any value other than `0.0`. ([#4758](https://github.com/kornia/kornia/pull/4758))
- dilation, erosion and the operations built on them (opening, closing, gradient, top_hat, bottom_hat) update the shift engine's running maximum/minimum in place when the call records no gradient, instead of allocating a new output-sized tensor at every kernel position. This is about 1.1-1.5x faster on CPU for a batch of eight 256 x 256 images, about neutral for a single small image, and unchanged on MPS; outputs are identical. Calls that record a backward graph, carry a forward-mode AD tangent on the image or the structuring element, run under torch.func transforms, or are traced or scripted keep the out-of-place reduction. ([#4759](https://github.com/kornia/kornia/pull/4759))
- `find_essential` and `run_5point` no longer abort with `torch._C._LinAlgError` when a
  correspondence set makes the 10x10 elimination matrix exactly singular, as a set of points all at
  the origin does. Such a sample has no solution and now returns `NaN` for all 10 candidates, like
  any other sample without candidates. Whether another degenerate set is exactly singular depends on
  the platform's LAPACK. Every other sample goes through the same solve as before and its result is
  unchanged bit for bit, including when a singular sample shares its batch, so a singular minimal
  sample inside `RANSAC(model_type="essential")` is dropped like any other sample without candidates
  instead of raising out of `forward`. Fixes #4765. ([#4766](https://github.com/kornia/kornia/pull/4766))
- Fix augmentation API documentation examples for AutoAugment, RandAugment, and TrivialAugment, and correct the ManyToManyAugmentationDispather name in the augmentation container metadata. ([#4769](https://github.com/kornia/kornia/pull/4769))
- The conventions pages are trimmed to kornia scope: the augmentation, randomness and serialization sections are condensed with every tracked defect link kept, `RandomEqualize` and `RandomClahe` are documented as raising out of range on every device, the ARCore camera frame is merged into the OpenGL row whose `camtoworld_graphics_to_vision_*` converters apply to it, and the Direct3D row and the upstream-quote appendix are removed. ([#4786](https://github.com/kornia/kornia/pull/4786))
- `geometry.transform` docstrings: removed the false `crop_by_boxes` note claiming a source box smaller than the destination box raises, and documented three known defects with their issues: `homography_warp(normalized_homography=False)` ignores `mode` and `align_corners` (#4772), `remap` resamples under an identity pixel map at its default `align_corners` (#4504), and `warp_perspective` and `warp_perspective3d` warp a `float64` input through a `float32` grid (#4776). ([#4788](https://github.com/kornia/kornia/pull/4788))
- The `kornia.morphology` docstrings are trimmed to what a caller needs: `dilation` carries the one convention block and lists the known defects (#4734, #4735, #4762) in one place, the other six functions refer to it, and an out-of-range `reflect` or `circular` pad is documented as raising an error rather than a `RuntimeError`, since that error comes from torch. The morphology table on the Conventions & Pitfalls page now holds the scipy, scikit-image and OpenCV mapping, including the kornia calls equal to their dilation, erosion, opening and closing and the `replicate` -> `nearest` and `circular` -> `wrap` border names. ([#4789](https://github.com/kornia/kornia/pull/4789))
- `solve_pnp_dlt`'s point-count, dtype and `svd_eps` checks now name the offending argument (for example "world_points must hold at least 6 points (N >= 6), got N = 5.") instead of a bare "Validation condition failed". The convention blocks of `kornia.geometry.bbox`, `kornia.geometry.boxes`, `kornia.geometry.depth`, `kornia.geometry.camera`, `kornia.geometry.calibration` and `kornia.sensors.camera` are trimmed to kornia's own contract, with the pixel-centre convention stated once on `PinholeCamera`, and a `torch.where:` typo for "where:" is fixed in the docstrings of `harris_response`, `gftt_response`, `hessian_response`, `BinaryFocalLossWithLogits`, `ssim3d`, `SSIM3D`, `solve_pnp_dlt` and `homography_i_H_ref`. ([#4790](https://github.com/kornia/kornia/pull/4790))
- The `kornia.augmentation` Convention blocks outside the 2D intensity classes are shorter and state each rule once. `RandomEqualize3D` now documents its out-of-range check as raising on every backend; the old text said MPS skipped it. Each known defect is stated with its tracking issue, including the new #4775, #4776, #4777, #4787, #4801, #4802, #4804, #4805, #4809 and #4815. ([#4791](https://github.com/kornia/kornia/pull/4791))
- `kornia.geometry.conversions` docstrings no longer claim that the deprecated aliases break `-W error::DeprecationWarning` (fixed in #3956), state the float64 axis-angle round-trip accuracy correctly (roundoff, about `2e-10` on the low-angle branch), and document a defect: `Rt_to_matrix4x4` and `camtoworld_to_worldtocam_Rt` handle an `R`/`t` batch-size mismatch differently (#4774). ([#4792](https://github.com/kornia/kornia/pull/4792))
- Corrected the 2D intensity augmentation docstrings and the `kornia.enhance` `AdjustHue`, `AdjustContrastWithMeanSubtraction`, `AdjustSaturationWithGraySubtraction` and `AdjustBrightnessAccumulative` summaries: `RandomBrightness`, `RandomContrast` and `RandomSaturation` no longer claim to match PIL and torchvision (their formulas differ, and `ColorJitter` is the class whose steps use torchvision's), `RandomHue`, `RandomGamma` and the `enhance` modules state the mapping they do share with torchvision, the stale MPS and raw-error notes on `RandomEqualize` and `RandomClahe` are gone, and the convention blocks are trimmed to kornia's own behaviour. Newly found defects are linked as known issues: #4781, #4782, #4783, #4784, #4785, #4807, #4810, #4811 and #4813. ([#4793](https://github.com/kornia/kornia/pull/4793))
- `find_homography_dlt` no longer hangs on CUDA when a four-point sample contains NaN or infinite values: the minimal LU path keeps non-finite entries out of `torch.linalg.qr` and returns a NaN homography for the affected batch elements, as on CPU. ([#4794](https://github.com/kornia/kornia/pull/4794))
- `warp_perspective`, `warp_perspective3d` and `homography_warp3d` build their sampling grid in the input dtype
  instead of `float32`. A `float64` identity warp, including `RandomPerspective` and `RandomPerspective3D` with
  `distortion_scale=0`, now reproduces its input to `float64` roundoff, as `warp_affine` does. `float32` and
  half-precision grids are unchanged at the default `float32` default dtype; under
  `torch.set_default_dtype(torch.float64)` they are no longer built in `float64` and rounded, so they now match
  the result at the default. ([#4795](https://github.com/kornia/kornia/pull/4795))
- The Conventions page and the `remap` docstring now link to `kornia.geometry.grid.create_meshgrid` instead of rendering its name as plain text. ([#4797](https://github.com/kornia/kornia/pull/4797))
- `solve_quartic` now takes its cubic fallback for `float16` and `bfloat16` inputs whose leading coefficient is below
  `1e-6` in magnitude, the tolerance `float32` uses, since half-precision quartics are solved in `float32`. `float16`
  used to round the fallback tolerance to 0: a zero leading coefficient returned NaN and one below `1e-6` returned an
  infinite fourth root. `bfloat16` lost every root of `x^3 - 6x^2 + 11x - 6` behind a leading coefficient of `1e-9`.
  Fixes #4498. ([#4822](https://github.com/kornia/kornia/pull/4822))
- `kornia.augmentation.__all__` now lists `RandomClahe` and `RandomJPEG`, so `from kornia.augmentation import *` includes them. Both classes were importable by name before but had never been in `__all__`. ([#4825](https://github.com/kornia/kornia/pull/4825))
- Clarify that `draw_point2d` and `draw_line` modify the input image in place and return it for convenience. ([#4826](https://github.com/kornia/kornia/pull/4826))
- `ColorJitter` skips a step whose factors are all neutral again, as it did before v0.9.0rc1, instead of computing every step and discarding the neutral ones: `ColorJitter(brightness=0.2)` accepts one- and four-channel images again, and an eager call no longer pays for the neutral colour-space round trips. Outputs for inputs that were accepted before are unchanged (#4813). ([#4828](https://github.com/kornia/kornia/pull/4828))
- `depth_from_plane_equation` floors a positive `eps` at float16's smallest subnormal, `2**-24`. The default
  `eps=1e-8` rounded to zero in float16, so a ray parallel to the plane returned `inf`, or `nan` for a plane through the
  camera centre; it now returns `d / 2**-24`, which float16 can represent only for `|d|` below about `3.9e-3`. Larger
  offsets still return `inf` in float16 unless a larger `eps` is passed. ([#4830](https://github.com/kornia/kornia/pull/4830))
- `crop_by_boxes` and `crop_and_resize` now accept a box with a size-1 axis. Its vertices are collinear, so solving the perspective system from them returned a matrix of `NaN`; such a box now gets the matrix built from the box extents, with a plain translation on the size-1 axis. Boxes the solve already handled keep the same matrix. Resizing a size-1 axis to a larger size, or a larger axis down to size 1, still returns `NaN` (zeros with `mode="nearest"`), because no invertible matrix maps one onto the other. ([#4834](https://github.com/kornia/kornia/pull/4834))
- `solve_cubic` now returns the real root of a cubic whose depressed form is `t^3 = 2R` with `R < 0` (for example `x^3 + 1`) instead of NaN. `solve_quartic` inherited the NaN through its cubic fallback and through its resolvent cubic: `x^4 - 3x^2 - 0.75` returned four NaNs instead of its real roots `±1.798`. The gradient of that `Q == 0` root also used to drop the root's dependence on `Q`, so for `x^3 - 1` the derivative with respect to the linear coefficient was 0 instead of -1/3; it now matches the implicit derivative. (#4832) ([#4837](https://github.com/kornia/kornia/pull/4837))
- PlanckianJitterGenerator keeps its index sampler in float32 when the augmentation is moved to float16 or bfloat16, so the sampled Planckian table index cannot go one past the end. On MPS, half-precision `torch.rand` returns exactly 1.0 for about 1 in 4096 (float16) or 1 in 512 (bfloat16) draws, which made `RandomPlanckianJitter` raise an out-of-bounds index error. float32 and float64 draws are unchanged; for the same seed, float16 and bfloat16 pipelines now draw some different indices than before. (#4553) ([#4839](https://github.com/kornia/kornia/pull/4839))
- `confusion_matrix` and `mean_iou` return an empty `(0, K, K)` / `(0, K)` result for a zero-length batch instead of raising `RuntimeError` on `view(0, -1)`. ([#4845](https://github.com/kornia/kornia/pull/4845))
- `find_essential` and `run_5point` no longer abort when a degenerate sample makes the companion matrix
  of the 5-point solver non-finite. That happens at float32 when the elimination system is
  ill-conditioned without being exactly singular and its solution overflows the determinant
  polynomial, or when a finite polynomial overflows on division by its leading coefficient.
  `torch.linalg.eigvals` rejected that matrix for the whole batch: with a `RuntimeError` on macOS, and
  by crashing the process inside MKL on Linux x86. Such a sample has no usable roots and now returns
  `NaN` for all 10 candidates, like a singular one. Every other sample is unchanged bit for bit,
  including one that shares its batch. Which inputs reach this path depends on the platform's LAPACK. ([#4846](https://github.com/kornia/kornia/pull/4846))
- `ColorJitter`, `RandomGaussianBlur` and `RandomGaussianIllumination` now pickle and pass through `torch.save`
  after their own `compile()`. Unpickling compiles the restored module again with the same arguments. (#4849) ([#4849](https://github.com/kornia/kornia/pull/4849))
- `draw_line` accepts a 0-d scalar color for single-channel images instead of raising `IndexError` on `color.size(0)`. ([#4850](https://github.com/kornia/kornia/pull/4850))
- `RandomResizedCrop(resample="nearest")` no longer raises in the default slice mode. Slice mode now ignores
  `align_corners` for nearest resampling, for images and masks. (#4852) ([#4852](https://github.com/kornia/kornia/pull/4852))
- `pixi run -e cuda install` now installs the CUDA 12.1 PyTorch build into the project `.venv` that the test tasks use,
  and CUDA tasks no longer re-sync it back to the locked PyTorch. The task used to pass the CUDA index by name, which uv
  does not resolve, so it installed PyPI's default build into Pixi's `.pixi/envs/cuda` and left `.venv` on the locked build. ([#4858](https://github.com/kornia/kornia/pull/4858))
- `AugmentationSequential` now normalizes a string or int `resample` in `extra_args` as the augmentation
  constructors do. It used to raise `AttributeError` wherever the mask was resampled. (#4859) ([#4859](https://github.com/kornia/kornia/pull/4859))
- `find_essential` and `run_5point` now return correct input gradients for fewer than 9 points. The
  solver's null space comes from right singular vectors that `torch.linalg.svd` gives no gradient to
  when there are fewer than 9 points, so the gradient to the correspondences was dropped: exactly zero
  for the 5-point minimal sample, and wrong for 6 to 8 points. The null space is now differentiated as
  a subspace, and the gradients agree with finite differences. The forward result is unchanged, and so
  are the gradients for 9 points or more. ([#4860](https://github.com/kornia/kornia/pull/4860))
- `warp_frame_depth` now raises when `image_src` and `depth_dst` have different heights or widths, instead of resampling the whole image onto the depth grid. ([#4861](https://github.com/kornia/kornia/pull/4861))
- `draw_convex_polygon` returns the input image unchanged for an empty `(B, 0, 2)` polygon batch instead of raising `IndexError` when closing the polygon loop. ([#4895](https://github.com/kornia/kornia/pull/4895))
- `Boxes.from_tensor` accepts nested numeric lists shaped like `(N, 4)` and an empty Python list instead of raising `AttributeError` / `ValueError`. ([#4896](https://github.com/kornia/kornia/pull/4896))
- `solve_cubic` no longer returns NaN for `float16` cubics whose `Q^3` underflows to 0 while `R` is 0, such as `x^3 - 0.003x` or `x^3 + 0.003x`. Half-precision cubics are now solved in `float32` and the roots are cast back to the input dtype, as `solve_quartic` already did for its Ferrari path. This also fixes the `bfloat16` reference quartic that went through the cubic fallback. (#4856) ([#4898](https://github.com/kornia/kornia/pull/4898))
- `KRt_from_projection` returns a proper rotation and the same `K`, `R`, `t` for every nonzero scale of `P`, including `-P` and a `P` scaled below the old `eps`, which is now unused (#4864). `triangulate_points` returns NaN for a correspondence at infinity instead of a finite point at a roundoff-set distance (#4865); `solver="cofactor"` normalises in the compute dtype, so pixel-scale float16 input is finite (#4863), and returns the `svd` point instead of averaging two normalised sub-system null vectors: it refines the longest of the four through the adjugate of the DLT matrix. The average was up to 3.5 units off on a noise-free rectified pair and 144 on a pair rolled by 90 degrees, 3 to 6 units off on every point of a rectified, vertical or rolled pair with 0.5 px of noise (0.1 for `svd`), and a roundoff direction for zero baseline, which now gives NaN (#4900). ([#4901](https://github.com/kornia/kornia/pull/4901))
- `RANSAC(score_type="msac")` accepts models again. Acceptance and confidence stopping treated the MSAC score as an inlier
  count, and a hypothesis was kept only when that score exceeded the minimal sample size. The old score subtracted the
  truncated squared residuals from `N`, so with a 2-pixel threshold it usually fell below that cutoff and the estimator
  returned the all-zero failure matrix: eight-point fundamental estimation on 60 IMC phototourism pairs failed for 87% of
  SIFT and all XFeat pairs. With thresholds below 1 the score could instead exceed the inlier count and stop sampling too
  early. The score is now `sum(1 - min(e / inl_th**2, 1))`, accumulated in at least float32, and the inlier count is
  tracked separately for acceptance and stopping, so those pairs no longer fail. Early stopping is checked after every
  batch, using the current best model's support after local optimization. Zero and nonfinite candidate matrices are
  rejected, since a zero fundamental matrix has zero Sampson error for every match. A model needs more inliers than its
  minimal sample for every score, so five-point essential estimates with six or seven inliers, fewer than the eight-point
  polisher needs, are kept. Local optimization keeps a full-inlier refit
  that ties the score instead of discarding it, so RANSAC scoring no longer returns the less precise minimal-sample model
  when support does not grow. The essential-matrix mask is recomputed after projection onto the essential manifold, and
  a projected model left with five inliers or fewer is reported as the failure result. ([#4902](https://github.com/kornia/kornia/pull/4902))
- `enhance.equalize_clahe` and `augmentation.RandomClahe` now raise kornia's `ValueError` when either image
  axis is not larger than its `grid_size` entry, and the message names the smallest image the grid admits.
  An image exactly as large as the grid, such as `8 x 8` at the default `grid_size=(8, 8)`, previously passed
  the size check and failed inside the reflect padding with torch's raw `RuntimeError`. Closes #4783. ([#4907](https://github.com/kornia/kornia/pull/4907))
- `draw_convex_polygon` now draws a one-vertex polygon as that point, leaves the image of an empty polygon in a list unchanged, and accepts an empty list with a zero-batch image, instead of raising PyTorch errors. The edge slope no longer divides by `dy + 1e-12`: in float16 that epsilon underflowed and left the scanline through a repeated vertex (including the padding of a shorter polygon in a list) unfilled, and in float64 it dropped pixels whose centre lies exactly on a sloped edge. ([#4909](https://github.com/kornia/kornia/pull/4909))
- `draw_point2d`, `draw_rectangle` and `draw_convex_polygon` now accept a 0-d color such as `torch.tensor(9.0)`, like `draw_line`, instead of raising `IndexError` or `ValueError`. ([#4910](https://github.com/kornia/kornia/pull/4910))
- `solve_quadratic`, `solve_cubic` and `solve_quartic` now solve a polynomial with a zero leading coefficient as the lower-degree polynomial it is. Before, `solve_quadratic` returned `nan`/`inf` for a linear equation, `solve_cubic` returned `1.0` for every linear equation and zeros for `bx² + d`, and `solve_quartic` inherited both. (#4873) ([#4913](https://github.com/kornia/kornia/pull/4913))
- `decompose_essential_matrix_no_svd` now returns rotations for every element of a batch. It used to divide each
  rotation by `b · b` summed over the whole batch, so each rotation of a batch of two or more came back scaled by its
  matrix's share of the batch's squared Frobenius norm: by `1 / B` when all `B` matrices have the same norm, such as the
  unit-norm candidates of `find_essential`. A batch of one is unchanged bit for bit. (#4915) ([#4915](https://github.com/kornia/kornia/pull/4915))
- `AugmentationSequential.show()` and `.save()` now work after a forward call that returns tensors (the default
  `output_type="pt"`). The container cached the raw forward output, so `.show()` raised `RuntimeError: Can't call
  numpy() on Tensor that requires grad` for an input that requires gradients, a `TypeError` for a CUDA or MPS one,
  and `AttributeError: 'list' object has no attribute 'shape'` for a call with several data keys. It now caches a
  detached view of the augmented image alone, which no longer holds the autograd graph, and `.show()` / `.save()` move
  it to the CPU when they render it, so the forward pass still makes no device-to-host copy. A call whose data keys
  carry no image caches nothing, and `ImageModule.save()` reports an empty cache with the same `ValueError` as
  `.show()` instead of `AttributeError: 'NoneType' object has no attribute 'shape'`. Closes #4835. ([#4918](https://github.com/kornia/kornia/pull/4918))
- `solve_quadratic` no longer loses the root of smaller magnitude to cancellation when `|4ac|` is much smaller than `b^2`: it uses `q = -(b + sign(b) sqrt(D)) / 2` and returns `c / q` and `q / a` in the previous slot order. `[1e-8, 2, -6]` gave `[0, -2e8]` in float32 and now gives `[3, -2e8]`, and the root's gradient is correct too. ([#4919](https://github.com/kornia/kornia/pull/4919))
- `solve_quartic` decides whether a row is really a cubic relative to the row's scale: the leading-coefficient tolerance for the cubic fallback (`1e-6`, or `1e-12` in float64) is multiplied by the largest coefficient magnitude of the row, capped at 1. `(x - 1)(x - 2)(x - 3)(x - 4)` scaled by `5e-7` in float32 (or `1e-13` in float64) now returns `1, 2, 3, 4` instead of a single wrong root. Rows whose largest coefficient is 1 or more keep the previous absolute tolerance. (#4905) ([#4921](https://github.com/kornia/kornia/pull/4921))
- `So3.log`, `So3.right_jacobian`, `So3.left_jacobian`, `Se3.exp` and `Se3.log` now keep their digits for small
  rotations (#4897). `So3.log` measures the angle with `atan2` instead of `2 * acos(w)`, which in float32 returned exactly
  `[0, 0, 0]` for every rotation below 1e-4 rad and was 2% short at 1e-3 rad. The Jacobians and the `Se3` `V` and `V^-1`
  matrices evaluate `(1 - cos theta) / theta**2`, `(theta - sin theta) / theta**3` and `(1 - (theta / 2) cot(theta / 2)) /
  theta**2` from their series below 0.5 rad (0.2 rad in float64), so they are the identity instead of `nan` at
  `theta = 0` and `Se3.exp(v).log()` returns `v` for rotations below 1e-4 rad in float32 (1e-8 rad in float64) instead of
  dropping the rotation-coupled part of the translation. ([#4941](https://github.com/kornia/kornia/pull/4941))
- `Quaternion.slerp` takes the shorter arc between the two rotations whatever the signs of the stored quaternions (it used to go the long way round when their dot product was negative). (#4944) ([#4945](https://github.com/kornia/kornia/pull/4945))
- `compose_transformations` and `relative_transformation` now broadcast a batch of one against a batch of `N` in
  either argument. Before, a `(1, 4, 4)` first argument with an `(N, 4, 4)` second argument raised a torch
  `RuntimeError`, while the reverse order worked. Batches of two different sizes other than one now raise a
  `ValueError` instead of a torch `RuntimeError`. Fixes #4933. ([#4946](https://github.com/kornia/kornia/pull/4946))
- `So2`, `Se2`, `Se3` and `ParametrizedLine` built from a tensor that requires grad keep that tensor, so the gradient reaches it; they used to wrap it in a new `nn.Parameter` and leave the caller's `.grad` at `None`. Such a tensor is now registered as a buffer: it no longer appears in the module's `.parameters()`, but `state_dict()`, `load_state_dict()` and `.to()` still reach it. A tensor that does not require grad still becomes a parameter. (#4943) ([#4947](https://github.com/kornia/kornia/pull/4947))
- `transform_points` now applies an unbatched transform to point sets of any size. ([#4958](https://github.com/kornia/kornia/pull/4958))
- `ImageSequential.show()` / `.save()` and `AugmentationSequential.show()` / `.save()` now work after a call with
  `output_type="numpy"` or `"pil"`. The containers cached the converted output, so `.show()` raised `TypeError: Input
  tensor must be a PyTorch tensor.` for NumPy and `AttributeError: 'list' object has no attribute 'shape'` for PIL. They
  now cache the augmented tensor before converting the output; the returned output is unchanged. Closes #4956. ([#4959](https://github.com/kornia/kornia/pull/4959))
- `Se2.exp` and `Se2.log` now keep the translation at and near `theta = 0` (#4924). Both guarded their 0/0 coefficients
  with a fallback of 0 where the limit is 1, so `Se2.exp([1, 2, 0]).t` was `[0, 0]`, the log of every pure translation was
  `[0, 0, 0]`, and just above `theta = 0` the closed forms cancelled: `Se2.exp(v).log()` was off by 2.0 at `theta = 1e-4`
  in float32 (2e-2 in float16, 1e-8 in float64). `Se2.log` now evaluates `(theta / 2) cot(theta / 2)` through the
  cancellation-free `So3` coefficients, and `Se2.exp` evaluates `sin(theta) / theta` and `(1 - cos(theta)) / theta` through
  them below 0.5 rad and as `sin(theta) / theta` and `2 sin(theta / 2)^2 / theta` above it, which also keeps `exp(v).t`
  to a few ulps where it is nearly zero (`theta` near a nonzero multiple of `2 pi`). The gradients at the identity are the
  true ones instead of 0 and `nan`. For a `v` that does not require grad, `Se2.exp(v).t` is now an `nn.Parameter` like
  `Se3.exp(v).t`, instead of a tensor whose autograd graph led to the hidden `So2` parameter. The `So3` coefficient
  series are now evaluated on 0 above their switch, so the float16 gradients of `Se3.exp`, `So3.left_jacobian` and
  `So3.right_jacobian` are finite from 50 to 255 rad instead of `nan` (#4965). ([#4960](https://github.com/kornia/kornia/pull/4960))
- A weight of exactly 0, the documented way to drop a correspondence, now gets its derivative from above in `find_fundamental`, `run_8point` and the weighted `normalize_points`, and so in the weighted `find_homography_dlt` and `find_homography_lines_dlt`, on every supported torch. `find_fundamental` and `run_8point` used to return `NaN` for it on torch 2.5.1 and 2.9.1, and on torch 2.14 `clamp_min(0)` masked the gradient to 0 in `run_8point` and dropped the Hartley-normalization part of it in `normalize_points`. ([#4961](https://github.com/kornia/kornia/pull/4961))
- `So3.exp`, and with it `Se3.exp`, now switches to its small-angle series below a fixed 0.5 rad instead of below
  `finfo(dtype).eps * 1e3`, which is 7.8 rad in bfloat16 and 0.98 rad in float16 (#4928). Every bfloat16 `exp` used the
  two-term series `0.5 - theta**2 / 48` for `sin(theta / 2) / theta`, so at `theta = 3` the quaternion had norm 0.94 and
  the rotation matrix was off by 0.23 in an entry; the quaternion is now within about one bfloat16 ulp (at most 5.5e-3
  for angles up to pi), and in float16 at `theta = 0.97` its error drops from 2.8e-4 to 9.1e-5. The series now runs
  through `theta**10`, so the float32 and float64 outputs move by at most 1.5e-8 and 2.8e-17 below 0.5 rad, a rounding
  difference, and do not change above it. ([#4962](https://github.com/kornia/kornia/pull/4962))
- `Se3.exp` and `Se3.log` now have the right gradient at the identity (#4953). Both fell back to `t = upsilon`
  (`upsilon = t`) at `omega = 0`, which is the right value since `V(0) = I` but does not depend on `omega`, so autograd
  returned `d t / d omega = 0` there instead of `-0.5 [upsilon]_x`, and `d upsilon / d q_vec = 0` for `log` instead of
  `[t]_x`. The coefficients of `V` and `V^-1` have been finite at `theta = 0` since #4941, so `V @ upsilon` is now taken
  for every element. The forward values are unchanged. ([#4963](https://github.com/kornia/kornia/pull/4963))
- `_so3_small_angle_coefficients` now divides its closed forms by `theta` one factor at a time instead of by `theta**2`
  and `theta**3`, which overflow float16 above 256 and 40.3 rad (#4965). From 41 rad the `[omega]_x^2` coefficient was
  0, so the float16 `Se3.exp(v).t` was off by 0.18 and `So3.right_jacobian` and `So3.left_jacobian` by 0.79 up to 250
  rad; they are now within about 2 float16 ulp there. The gradient of the coefficients, `nan` from 256 rad, is finite.
  Above the series switch the float32 and float64 coefficients move by at most about 2 ulp; below it they are unchanged. ([#4967](https://github.com/kornia/kornia/pull/4967))
- `kornia.geometry.euclidean_distance` now returns the exact Euclidean distance. The previous form
  added `eps` (1e-6) inside the square root, which biased every result upwards and floored it at
  `sqrt(eps)` = 1e-3: coincident points read 0.001 and a true distance of 1e-4 read 1.005e-3
  (+905 %). The singular point is guarded with `torch.where` instead, so coincident points return
  exactly 0 with a finite (zero) gradient, as `torch.linalg.norm` does. The `eps` argument is now
  unused and is kept for backward compatibility. ([#4968](https://github.com/kornia/kornia/pull/4968))
- `RandomClahe` and the per-sample CLAHE path now compute their clip thresholds correctly when the clip tensor is on MPS
  (#4970). The limits were moved to the CPU in float64 with a single `.to("cpu", torch.float64)`, which returns zeros
  on torch 2.14, so every tile histogram was clipped at one count (a mean error of 0.16 on a 256x256 image), and raises
  `TypeError` on torch 2.5.1. Parameters drawn by the default generator stay on the CPU and were not affected. ([#4971](https://github.com/kornia/kornia/pull/4971))
- `So3.exp` now evaluates `cos(theta / 2)` and `sin(theta / 2) / theta` below the 0.5 rad switch as series in
  `theta**2 = v . v` instead of on `theta = |v|`, which has no second derivative at `v = 0`, so the Hessian of `exp` at
  the identity is finite (`-I / 4` for the real part) instead of `nan` and `gradgradcheck` passes there (#4966). Above
  the switch the output is bitwise unchanged in every dtype; below it the quaternion moves by at most 1 ulp. ([#4972](https://github.com/kornia/kornia/pull/4972))
- `So2.random` and `Se2.random` now draw a unit rotation with a uniform angle on [-pi, pi) (#4930). They built
  `z = complex(rand, rand)`, so `|z|` ranged over (0, sqrt 2), `matrix()` was a rotation scaled by `|z|` with
  `det = |z|**2` down to 3.5e-6, and every angle was in [0, pi / 2]. The random values under a fixed seed change. ([#4973](https://github.com/kornia/kornia/pull/4973))
- `point_line_distance` now divides by the weight of a homogeneous `(*, N, 3)` point, `|ax + by + cw| / (|w| |(a, b)|)`,
  and returns `inf` for a point at infinity (#4935). The weight was ignored, so a point with `w != 1` got the distance
  of `(x, y)` instead of `(x / w, y / w)`, and `left_to_right_epipolar_distance` and `right_to_left_epipolar_distance`
  inherited it for `(*, N, 3)` input. 2-D points and `w = 1` are unchanged. ([#4975](https://github.com/kornia/kornia/pull/4975))
- `RANSAC(prosac_sampling=True)` always ran the whole `batch_size * max_iter` budget. It now stops with PROSAC's
  termination-length test (Chum and Matas, 2005): a ranked prefix certifies the incumbent when its support there is
  non-random and the draws it needs for `confidence` fit inside the prefix's growth interval, with OpenCV USAC's guards
  against short or weakly supported prefixes; the whole set is always a candidate, so PROSAC never runs longer than
  uniform sampling would. `confidence=1` still runs the whole budget. `find_homography_dlt` with the default
  `solver="lu"` computed the gauge null vector of a four-point sample with `torch.linalg.qr`, which has no batched CUDA
  kernel and factored a batch of 2048 minimal samples one matrix at a time in about 250 ms. On CUDA it now uses the
  batched SVD (about 1 ms), so `RANSAC("homography")` batches on GPU are up to 7x faster; CPU results are unchanged.
  `find_essential` on CUDA computed the roots of Nistér's degree-ten polynomial with `torch.linalg.eigvals`, which has
  no batched CUDA kernel and factors the companion matrices one cusolver call at a time (about 3.6 ms each), so
  `RANSAC("essential")` spent 6.4 s per batch of 2048 samples. The eigenvalues are now computed on the host with LAPACK
  for CUDA inputs (about 15 ms for that batch, transfers included), which makes the solver about 150x faster on the GPU;
  results are unchanged to roundoff. `RANSAC("homography_from_linesegments")` local
  optimization re-weighted segments by the length-scaled residual of `line_segment_transfer_error_one_way`, which
  down-weighted long segments (#4867); it now re-weights by the perpendicular distance in pixels that the score uses. ([#4979](https://github.com/kornia/kornia/pull/4979))
- `average_quaternions` now normalises its members before averaging and raises `ValueError` for a negative weight or
  weights that are all zero (#4974). A member's squared norm used to act as an extra weight (one stored as `3 q` counted
  9 times), a negative weight extrapolated past a member, and an all-zero weight divided by zero. Unit members with
  non-negative weights give the same rotation to rounding. ([#4980](https://github.com/kornia/kornia/pull/4980))
- `Quaternion.polar_angle` is computed as `atan2(|v|, w)` instead of `acos(w / |q|)`: the same angle up to rounding (at most 2.6e-6 rad on random float32 unit quaternions; angles below `sqrt(eps)`, which used to come out as exactly 0, are now resolved), and its gradient at the identity and at every other quaternion with a zero vector part is zero instead of `nan`. (#4927) ([#4981](https://github.com/kornia/kornia/pull/4981))
- `RandomMotionBlur` now raises a `ValueError` naming the class, the drawn kernel, the border type and the input shape when the image is too small for `border_type="reflect"` or `"circular"`, as `RandomBoxBlur` and `RandomGaussianBlur` already do, instead of letting a raw torch padding error out. The sizes it accepts are unchanged. ([#4982](https://github.com/kornia/kornia/pull/4982))
- `RandomTransplantation` and `RandomTransplantation3D` now raise a kornia error naming the mask when a call has to draw the parameters without one, adding the key count when the inputs do not match `data_keys`, instead of Python's `IndexError` or `list.index` `ValueError`. ([#4983](https://github.com/kornia/kornia/pull/4983))
- `RandomRGBShift` now accepts a `(low, high)` pair (a tuple, a list or a 1-D tensor) for each `*_shift_limit` as the sampling range, as `albumentations.RGBShift` does, instead of failing at construction with a raw `TypeError` from negating the tuple. A number is still a non-negative half-width sampled from `[-limit, limit]`. ([#4984](https://github.com/kornia/kornia/pull/4984))
- `RandomPerspective` no longer returns a NaN transform matrix and output for an image with a spatial dimension of 1. That axis now gets a unit source extent and no corner offset, so its single row or column maps onto itself and only the other axis is warped; at `distortion_scale=0` the warp is the identity. Images larger than one pixel along both axes draw and warp exactly as before. ([#4985](https://github.com/kornia/kornia/pull/4985))
- `crop_by_indices` joins the per-row crops with one `torch.cat` when every row's slice already matches `size`, instead of a `copy_` per row (#4531): one copy kernel per batch instead of one per sample, with the same values and the same default memory format. `RandomCrop` in slice mode always takes this path; rows that need `shape_compensation` still go through the per-row loop, and MPS keeps the per-row copies, which are faster there. A list `size`, as `RandomCrop([h, w])` passes it, no longer resizes every row to its own size. ([#4987](https://github.com/kornia/kornia/pull/4987))
- `Boxes.translate` with an unknown `method` now raises `NotImplementedError` with a message that names the accepted
  values and repeats the rejected one, instead of an empty message. The exception type is unchanged. ([#4988](https://github.com/kornia/kornia/pull/4988))
- `Quaternion.slerp` now reads a `(B,)` tensor ratio as one ratio per quaternion, like `(B, 1)`. With `B == 3` it
  scaled the components of each rotation vector instead and returned a wrong result, and other `B > 1` raised (#4991). ([#4993](https://github.com/kornia/kornia/pull/4993))
- `Quaternion.__pow__` takes the axis `n = e_x` on the negative real axis (`w < 0`, zero vector part), where `theta = pi`
  and the axis is undefined, so a non-integer power there has norm `|q|**t` and angle `t * pi` and `(q**0.5)**2 == q`.
  `Quaternion([-2, 0, 0, 0]) ** 0.5` used to be the zero quaternion. Integer powers are unchanged. (#4955) ([#5003](https://github.com/kornia/kornia/pull/5003))
- `find_fundamental(..., method="7POINT")` / `run_7point` now return zero matrices for the two extra candidates when the 7-point cubic has one real root, instead of one rank-3 matrix repeated. ([#5006](https://github.com/kornia/kornia/pull/5006))
- `find_homography_lines_dlt` builds each segment's equations from its own start and end instead of from flattened points i and N + i, so line segments whose endpoints are not point correspondences give the right homography, `RANSAC("homography_from_linesegments")` finds a model on them, and a zero weight removes its segment (#4866). ([#5008](https://github.com/kornia/kornia/pull/5008))
- `NamedPose` now rejects a pose that is not an `Se2` or an `Se3` at construction with a `TypeCheckError` (an `So3`,
  `So2`, `Quaternion` or tensor used to be accepted and fail later in `.rotation` with `AttributeError`), and
  `NamedPose.__mul__` raises a `TypeError` that names the offending type when the right operand is not a `NamedPose` or
  holds the other group, instead of an `AttributeError` from inside the dispatch. `from_rt` and `from_matrix` are
  unaffected. (#5010) ([#5010](https://github.com/kornia/kornia/pull/5010))
- `solve_cubic` evaluates the closed form on each row scaled by an exact power of two to a unit root bound and scales
  the roots back, takes Cardano's `A + B` as `2R / (A^2 + B^2 + Q)` when `Q > 0` and `A^2 < 4.79 Q`, and drops the
  `|R| > 1e-16` floor that returned `-b/3` for `D > 0` rows with a small `R`, so a tiny leading coefficient keeps its
  finite root in float32 and float64 and the #4833 literal solves (#4914). A row whose largest root exceeds 16 times
  the scale of the other two keeps that root and gets the other two from Vieta's relations and `solve_quadratic`, so
  `1e-8 x^3 + (x - 1)(x - 2)` returns all three roots instead of two values that are not roots. Outside that case,
  `D <= 0` rows whose `Q^3` and `R^2` stayed within the dtype's range keep their bits, rows whose intermediates
  overflowed or underflowed are now solved, and `D > 0` rows can move at rounding level. ([#5024](https://github.com/kornia/kornia/pull/5024))
- `motion_from_essential_choose_solution` now selects each batch element's own best candidate pose independently. Previously, every element in a batched call silently reused the first element's chosen candidate index, which could return a wrong rotation, translation, and triangulated points for any batch (size > 1) where different elements' correct candidates differ. ([#5025](https://github.com/kornia/kornia/pull/5025))
- `fit_line` now returns the direction in the input's dtype for unweighted 2-D `float16` and `bfloat16` points.
  Its vertical-line fallback constant had no dtype, so `torch.where` promoted the direction to `float32` while the
  origin kept the input's dtype. ([#5026](https://github.com/kornia/kornia/pull/5026))
- `spatial_expectation2d`, and so `spatial_soft_argmax2d`, now builds its normalised grid in the input dtype. float64
  coordinates no longer carry float32 rounding error. float16 and bfloat16 still build it in float32. (#5027) ([#5027](https://github.com/kornia/kornia/pull/5027))
- `ParametrizedLine.squared_distance` and `distance` no longer go negative or return NaN for points on or near the line.
  They use the sum of squares of the perpendicular component again, row by row, instead of a cancelling identity. (#5028) ([#5028](https://github.com/kornia/kornia/pull/5028))
- The seven-point fundamental solver keeps rank-two candidates when both null-space basis matrices have nearly vanishing determinants, and its cubic polishing no longer displaces repeated roots through a nearly zero derivative. RANSAC's Levenberg-Marquardt pipeline filters unsupported hypotheses before selecting its best candidates and recomputes the returned inlier mask after converting the model to the input dtype, returning failure if rounding removes sufficient support. ([#5031](https://github.com/kornia/kornia/pull/5031))
- `Hyperplane.through` no longer documents a two-point 2D form. That form never worked: `Hyperplane.through(p0, p1)` always raised a bare `Validation condition failed` (a 2D normal cannot be stored in a `Vector3`), and the unreachable code would have used the line direction as the normal. A two-point call now raises a `BaseError` saying that three points are required. The undocumented, unexported helper `kornia.geometry.plane.normalized`, used only by that code, is removed. ([#5034](https://github.com/kornia/kornia/pull/5034))
- `conv_soft_argmax2d` and `conv_soft_argmax3d` now compile with `torch.compile(fullgraph=True)` when `temperature` is a tensor. Their positivity check read the tensor under compile, because `is_exporting()` is `False` inside `torch.compile` on recent torch; the check now runs only outside graph capture, as in `spatial_softmax2d`. ([#5035](https://github.com/kornia/kornia/pull/5035))
- Batched `ParametrizedLine.point_at`, `ParametrizedLine.projection`, `ParametrizedLine.intersect` and
  `Hyperplane.projection` now work row by row. Previously they broadcast a per-row step or distance against the
  coordinate axis: a batch of `B` lines in `D` dimensions returned rows mixed with each other when `B == D` and raised
  for most other batch sizes, and a `(2, 1)` plane batch returned a `(2, 2, 3)` projection. `point_at` also reads a
  `Scalar` step as its tensor and returns a `torch.Tensor`, as annotated (#5013). ([#5046](https://github.com/kornia/kornia/pull/5046))
- `save_pointcloud_ply` (ASCII) writes every float64 coordinate with its shortest round-trip representation instead of 9 significant digits, so `double` properties no longer lose precision past the ninth digit (UTM-scale metres were off by millimetres). ([#5048](https://github.com/kornia/kornia/pull/5048))
- CropGenerator, ResizedCropGenerator, CropGenerator3D and PatchMixGenerator keep their position sampler in float32 when the augmentation is moved to float16 or bfloat16, so a crop or patch start cannot land one past the last valid position. On MPS, half-precision `torch.rand` returns exactly 1.0 for about 1 in 4096 (float16) or 1 in 512 (bfloat16) draws, which made `RandomCrop(cropping_mode="resample")` blend a zero padding line into the crop and put `RandomCrop3D` boxes and `PatchMix` patches one pixel past the edge. float32 and float64 draws are unchanged; for the same seed, float16 and bfloat16 pipelines now draw some different positions than before, and `RandomResizedCrop` also some different crop sizes, because its scale draw comes from the same sampler. (#4553) ([#5050](https://github.com/kornia/kornia/pull/5050))
- `Keypoints.pad` and `Keypoints.unpad` accept the unbatched `(N, 2)` container the class documents and constructs. Both added the `(B, 1)` padding column to the `(N,)` coordinate view in place, which broadcast to `(1, N)` and raised `RuntimeError: output with shape [N] doesn't match the broadcast shape [1, N]`, so `RandomCrop` failed on `(N, 2)` keypoints with and without `padding`. They now shape the padding to the container, as `Boxes.pad` does since #4372, and match the singleton-batch result; an unbatched container carries one image, so a padding with more than one row is rejected. `padding_size` is also moved to the keypoints' device. Batched results are unchanged (closes #5021). ([#5051](https://github.com/kornia/kornia/pull/5051))
- `Hyperplane.through` no longer returns NaN gradients when `p2 - p0` and `p1 - p0` are orthogonal and of equal
  length: the SVD fallback, which runs on every row, now sees a constant matrix with distinct singular values in the
  rows that take the cross product, so its backward no longer divides by a zero singular-value gap (#5056). ([#5058](https://github.com/kornia/kornia/pull/5058))
- Crop starts no longer land one past the last valid position for `CropGenerator` and `CropGenerator3D` with a float16
  or bfloat16 `size` tensor, for `RandomResizedCrop` with a float16 or bfloat16 `scale` or `ratio` tensor, and for
  `RandomCrop`, `RandomResizedCrop` and `RandomCrop3D` moved to float64: the start draw is scaled and floored in the
  wider dtype and the integer start is cast afterwards. When the draw and the crop parameters share a dtype, the
  parameters are unchanged. With a float16 or bfloat16 tensor, the same seed also moves a few in-range starts, those
  whose draw crossed a start boundary when it was rounded to half precision, to the start of the unrounded draw (#5052). ([#5061](https://github.com/kornia/kornia/pull/5061))
- `RandomPerspective3D` now handles singleton spatial dimensions without a singular perspective solve. These axes get unit source extents and zero corner offsets, and their single slice, row or column maps onto itself. Zero distortion gives an identity transform (output identity still needs `align_corners=True` while #4503 is open). Non-singleton volumes and random draws are unchanged. ([#5071](https://github.com/kornia/kornia/pull/5071))
- `ImageRegistrator.register` applies `tolerance` within each pyramid level, so every level takes at least its first optimization step. Previously the first loss of a finer level was compared with the final loss of the coarser level, and when the two differed by less than `tolerance` that level stopped before any update (#5073). ([#5074](https://github.com/kornia/kornia/pull/5074))
- `nms2d`, `nms3d`, `NonMaximaSuppression2d` and `NonMaximaSuppression3d` now return 0 at every suppressed position, as
  their Returns sections state. The value output used to be `input * mask`, so a suppressed `±inf` or `NaN` came back as
  `NaN`, which `topk` ranks above every maximum, and a suppressed negative value came back as `-0.0` (now `+0.0`). The
  gradient reaching a suppressed position is 0 even when the incoming gradient there is not finite. (#5067) ([#5076](https://github.com/kornia/kornia/pull/5076))
- `find_essential` solves in float64 whatever the input dtype, so a float32 five-point sample no longer misses the true essential matrix (#4884); MPS, which has no float64, solves on the host and copies the candidates back. It now runs on MPS, where `torch.linalg.eigvals` raised `NotImplementedError` (#4528), and on float16 and bfloat16 inputs. On 42000 exact five-point samples with random motions it misses the true matrix by more than 1e-3 12 times, where it missed it 31 times. Its roots come from the host for every device, detached, with a Newton step that supplies their gradient, so `backward` no longer raises at a multiple root (#4903) nor returns NaN gradients for samples without parallax (#4952), and repeated calls on CUDA return identical results. A five-point sample whose constraints are rank deficient, such as one with a repeated correspondence, returns ten NaN candidates and a zero gradient. At 2048 samples it takes 29 ms instead of 40 ms on CPU and 21 ms instead of 28 ms on CUDA; at 256 samples it costs about the same on CPU and 1 ms more on CUDA. ([#5078](https://github.com/kornia/kornia/pull/5078))
- `Vector3.normalized` and `Vector2.normalized` return a zero vector, not NaN, for a float16 zero input, and its
  gradient is zero instead of inf; `ParametrizedLine.through` does the same for coincident points when value checks are
  disabled. They now share `normalize_quaternion`'s arithmetic, and a non-zero input keeps its `F.normalize` value bit
  for bit (#5062). ([#5084](https://github.com/kornia/kornia/pull/5084))
- `RandomPerspective` exported with the legacy ONNX tracer now keeps a unit source extent on a size-1 axis, matching eager mode instead of returning zeros. ([#5089](https://github.com/kornia/kornia/pull/5089))
- `rgb_to_grayscale` and `bgr_to_grayscale` on a uint8 image no longer wrap the weighted sum modulo 256 (white came back as 1 and pure red as 180), which also garbled `load_image(..., ImageLoadType.GRAY8)` of a colour file. The uint8 result now matches `cv2.cvtColor(..., cv2.COLOR_RGB2GRAY)` bit for bit. ([#5090](https://github.com/kornia/kornia/pull/5090))
- `kornia.color.rgb_to_yuv422`, `yuv422_to_rgb`, `RgbToYuv422` and `Yuv422ToRgb` accept an odd height when the
  width is even. 4:2:2 subsamples chroma along the width only, but the shape guard also required an even
  height and raised `ShapeError`. Results for even heights are unchanged; the 4:2:0 functions still require an
  even height and width. ([#5091](https://github.com/kornia/kornia/pull/5091))
- `save_pointcloud_ply`, `save_pointcloud_ply_binary`, `load_pointcloud_ply`, and
  `load_pointcloud_ply_binary` now accept `pathlib.Path` and other string-backed
  `os.PathLike` filenames, which raised `TypeError` before. As for a `str`, the `.ply`
  extension check ignores case. ([#5092](https://github.com/kornia/kornia/pull/5092))
- Speed up `RANSAC` with `local_optimization="lm"`, the default for homographies, fundamental and essential matrices. Hypothesis verification allocates fewer temporaries and, on CPU, is tiled at about a million residuals. The Levenberg–Marquardt stages, which run on the host for every input device, evaluate only the cost of their last trial instead of building an unused Jacobian, and the final refinement drops correspondences outside its inlier mask before building its equations. Five-point essential estimation evaluates its Newton correction with bounded cumulative products instead of general powers. Returned models can differ slightly, since rounding differences can change which refinement steps are accepted. ([#5095](https://github.com/kornia/kornia/pull/5095))
- `jpeg_codec_differentiable`, `JPEGCodecDifferentiable` and `RandomJPEG` now keep the DCT basis cache out of `torch.compile`, `torch.export` and dynamo ONNX tracing, and build eager cache entries outside inference mode. A first inference-mode call no longer breaks later image-gradient computations. Exporting the codec no longer leaves cached fake or functional tensors that break subsequent eager calls. Eager calls still reuse the basis per dtype and device. ([#5098](https://github.com/kornia/kornia/pull/5098))
- `RandomMosaic` with a list box input now gives each tile the padding count of the image it was drawn from. A selected
  sample used to take its box count from the wrong source images, so it could drop real boxes or return padding rows as
  boxes such as `[7, 0, 7, 0]`, from a direct call and from `AugmentationSequential`. Tensor box inputs are unchanged.
  (#4715) ([#5100](https://github.com/kornia/kornia/pull/5100))
- `AugmentationSequential` now calls the box and keypoint handlers of a custom `RigidAffineAugmentationBase2D`
  subclass with its recorded matrix, as it does for a geometric child. It used to return those annotations unchanged
  next to the moved image, while `transform_matrix` reported the move. Mask handlers of such a subclass now receive the
  matrix too instead of `transform=None`, and a registered box or keypoint key whose handler the subclass does not
  implement raises `NotImplementedError` instead of passing through. With an annotation key listed before `input` in
  `data_keys`, the container used to transform the annotation before the image call recorded the matrix, so a
  geometric child such as `RandomAffine` raised on the first call and warped the annotation with the previous call's
  matrix after it; the image is now transformed first, and the outputs keep the caller's order. (#4481) ([#5105](https://github.com/kornia/kornia/pull/5105))
- `RgbToGrayscale()` without `rgb_weights` now leaves the weights to `rgb_to_grayscale`, which picks them for the image
  dtype. A uint8 image used to come back all zeros, because the module's float32 default weights were cast to uint8 as
  `[0, 0, 0]`; it now gets the functional's uint8 result, which matches `cv2.cvtColor` (#5090). A float64 image used to
  get float32-rounded weights (up to 1.2e-8 off); outside TorchScript it now gets float64 weights. float32, float16 and
  bfloat16 outputs, and modules built with explicit weights, are unchanged (#5109). ([#5111](https://github.com/kornia/kornia/pull/5111))
- `JPEGCodecDifferentiable` and `jpeg_codec_differentiable` now build their DCT and IDCT bases with `math.pi` instead of
  the float32 tensor `kornia.constants.pi` (#5107). Strict `torch.export` on torch 2.5.1 and 2.9.1 rejected the codec
  because its forward referenced that module-level tensor, and torch 2.5.1's `torch.export.export` is strict by default.
  The float64 bases were only float32-accurate (the DCT basis was up to 5.65e-7 off the exact cosines), so float64
  outputs change, typically by less than 1e-6 for an image in [0, 1]. float32, float16 and bfloat16 outputs are
  unchanged. ([#5112](https://github.com/kornia/kornia/pull/5112))
- `RandomPerspective3D` traced with `torch.jit.trace` on a volume with a size-1 axis now matches eager mode instead of
  raising a singular solve. The tracer passes the sizes as 0-d tensors, and both size-1 rules used to apply only to
  Python integers: `PerspectiveGenerator3D` now gives a traced size-1 axis the unit source extent and zero corner offset
  of eager mode, and `RandomPerspective3D` keeps, in the traced graph, the corner order that holds a size-1 width in
  place. Eager parameters and outputs are unchanged (#5110). ([#5115](https://github.com/kornia/kornia/pull/5115))
- `AugmentationSequential` now dispatches box and keypoint data keys to `IntensityAugmentationBase2D` children so
  subclass overrides of `apply_transform_box` and `apply_transform_keypoint` are called alongside `apply_transform_mask` (#5113). ([#5119](https://github.com/kornia/kornia/pull/5119))
- The unimplemented `Keypoints` and `Keypoints3D` paths now raise `NotImplementedError` with a message naming what
  is unsupported instead of a bare one: building either class from a list of tensors,
  `to_tensor(as_padded_sequence=True)` on both, and `Keypoints3D.pad`, `unpad`, `transform_keypoints` and
  `transform_keypoints_`. `Keypoints3D.transform_keypoints` points at
  `kornia.geometry.linalg.transform_points(M, keypoints.data)`, and its `Args` block now documents `M` as
  :math:`(4, 4)` or :math:`(B, 4, 4)` rather than the 2D :math:`(3, 3)` shape. The paths remain unimplemented. ([#5122](https://github.com/kornia/kornia/pull/5122))
- `dice_loss` and `DiceLoss` sum float16 and bfloat16 inputs over the image in float32 and return the loss in the usual
  dtype. In float16 a sum above 65504 used to overflow, so the loss came back as 1.0 or NaN instead of its value, with
  `average="micro"` from 128 × 256 pixels on. ([#5123](https://github.com/kornia/kornia/pull/5123))
- `axis_angle_to_rotation_matrix` is now accurate to float64 rounding for rotations below `1e-3` rad. Its low-angle branch used to drop the `theta**2` terms of Rodrigues' coefficients, an error of `theta**3 / 6` per entry (`1.3e-10` just below the switch), so the result jumped at `theta = 1e-3` and disagreed with the quaternion route by the same amount. ([#5124](https://github.com/kornia/kornia/pull/5124))
- `tversky_loss` and `TverskyLoss` sum float16 and bfloat16 inputs over the image in float32, form the ratio there and
  return the loss in the input dtype. In float16 a pixel count above 65504 used to overflow, so the loss came back as 1.0
  from 256 × 256 pixels on, or NaN with `ignore_index` pixels; in bfloat16 a loss below about 4e-3 was rounded to a
  multiple of 2**-8. ([#5129](https://github.com/kornia/kornia/pull/5129))
- `AugmentationSequential` now processes `input` before annotations so nested geometric augmentations and auto-augment policies use the current call's transformation matrix, regardless of caller argument order. ([#5130](https://github.com/kornia/kornia/pull/5130))
- `PatchDominantGradientOrientation`, `LAFOrienter`, `SIFTDescriptor`, `DenseSIFTDescriptor`, `SIFTDescriptorFromPyramid`,
  `SIFTFeature`, `SIFTFeatureScaleSpace`, `LightGlueMatcher`, `RandomHue`, `ColorJiggle` and `ColorJitter` now use
  `math.pi` instead of the float32 tensor `kornia.constants.pi` in their forward arithmetic (#5127). Their float64 results
  used the float32 value of pi, 8.7e-8 above pi: orientation angles move by up to about 3e-6 rad and descriptor entries by
  up to about 1e-6, the orientations `LightGlueMatcher` wraps into [0, 2 pi) are no longer 1.7e-7 rad too large, and a
  hue shift of `f` turns is no longer `f * 1.7e-7` rad too large (with float64 hue factors in `ColorJiggle` and
  `ColorJitter`). float32 results are unchanged. On CPU, `PatchDominantGradientOrientation` used to round 2 pi to the half
  dtype before its bin-to-angle product: a third of its float16 angles move by up to 3.9e-3 rad and 2% of its bfloat16
  angles by up to 3.1e-2 rad, and the frames oriented by `LAFOrienter` and by the default patch backend of `SIFTFeature`
  and `SIFTFeatureScaleSpace` move with them. Every other float16 and bfloat16 result, and every half result on MPS, is
  unchanged. ([#5131](https://github.com/kornia/kornia/pull/5131))
- `ImageModule`, `kornia.core.ImageSequential` and `kornia.augmentation.ImageSequential` no longer copy every forward
  output to the CPU. On CUDA and MPS that was a blocking device-to-host copy on each call, made only for `.show()` and
  `.save()`. The cache now holds a detached reference on the output's device, as `AugmentationSequential` already does,
  and `.show()` / `.save()` copy on use; `disable_features=True` still skips the cache. ([#5133](https://github.com/kornia/kornia/pull/5133))
- The `GeometricAugmentationBase2D` Convention block no longer says the 3D augmentations accept an unbounded angle range: `RandomRotation3D`, `RandomAffine3D` and `RandomMotionBlur3D` raise past `±360` like the 2D classes. ([#5140](https://github.com/kornia/kornia/pull/5140))
- `LightGlueMatcher("sift")` now gives LightGlue each keypoint's scale and orientation in the convention of the COLMAP keypoints its weights were trained on: the orientation is the negated LAF orientation in (-pi, pi], and the scale is a sixth of the LAF scale, which is sigma for 6-sigma DoG frames such as `SIFTFeatureScaleSpace`'s. It used to pass the LAF scale and kornia's orientation in [0, 2pi). With OpenCV-layout descriptors, correct matches over the eight Oxford affine sequences rise by 1–2.4% for the kornia SIFT pipelines, and by up to 28% on bark, whose pairs rotate. ([#5145](https://github.com/kornia/kornia/pull/5145))
- `ssim`, `ssim3d`, `SSIM` and `SSIM3D`, and the `ssim_loss`, `SSIMLoss`, `ssim3d_loss` and `SSIM3DLoss` losses built
  on them, compute float16 and bfloat16 inputs in float32 and return the map in the input dtype. In float16 the products
  of the local moments overflowed for pixel-range data, so with `max_val=255.0` the map and the loss came back NaN (a
  constant image of value 24 or more compared with itself was NaN everywhere), and with `max_val` of 0.75 or less an
  all-zero region gave NaN because `C1 * C2` underflowed to zero; bfloat16 maps were up to 0.08 off on random
  32 x 32 images. The convolutions also run with autocast disabled now: under `torch.autocast` a float32 input used to
  give a map in the autocast dtype (bfloat16 on CPU, up to 0.07 off, and NaN for pixel-range data under float16
  autocast), and now gives a float32 map. ([#5148](https://github.com/kornia/kornia/pull/5148))
- Call-time `data_keys` passed to `AugmentationSequential.forward` or `.inverse` now apply to that call only. A call
  that raised, and an `inverse` call that succeeded, used to leave its keys as the container's default, so a later
  call without `data_keys` raised on an argument-count mismatch or sent each input through another key's handler.
  (#5136) ([#5149](https://github.com/kornia/kornia/pull/5149))
- `focal_loss` and `FocalLoss` now exclude ignored logits before log-softmax, so extreme finite logits (whose
  log-probability overflows) and infinite or NaN logits at ignored positions no longer turn the loss and its gradients
  into NaN. ([#5153](https://github.com/kornia/kornia/pull/5153))
- Fix `ImageModule.show()` / `.save()` and core `ImageSequential.show()` / `.save()` after requesting NumPy or PIL output by retaining a detached tensor cache before conversion. ([#5178](https://github.com/kornia/kornia/pull/5178))
- `otsu_threshold(x, return_mask=True)` now returns the comparison `x > threshold`, taken per image and channel against
  that plane's Otsu threshold and reshaped to the input, so a foreground pixel whose value is 0 or negative is `True`
  (#5173). Previously the mask was `result > 0` with `result = x * (x > threshold)`: on data whose upper level is 0 or
  below, for example values in {-1, 0} or {-2, -1}, the threshold separated the two levels but every mask pixel came out
  `False`. Only a negative threshold is affected: the pixels in `(threshold, 0]` are now `True`, and the mask is unchanged
  whenever the data are non-negative. The thresholded image is unchanged. ([#5182](https://github.com/kornia/kornia/pull/5182))
- `torch.compile(fullgraph=True)` now traces `RandomCrop` at `p=1`, `crop_by_indices` with `size` and bilinear, bicubic or nearest resampling, and the `ImageModule`, `ImageSequential` and `AugmentationSequential` call paths, including a fixed-order `ColorJiggle`. The containers no longer go through `functools.wraps` inside the traced call. Part of #5137. ([#5183](https://github.com/kornia/kornia/pull/5183))
- `edge_aware_blur_pool2d` and `EdgeAwareBlurPool2D` accept an even `kernel_size` and keep the input shape, instead of
  raising a `RuntimeError` about mismatched tensor sizes. ([#5184](https://github.com/kornia/kornia/pull/5184))
- `ImageModule`, `ImageSequential` and the augmentation containers no longer load a string argument as an image unless it is the first positional argument, or a keyword argument listed in `input_names_to_handle`. An option such as `mode="bilinear"` is now passed through unchanged even when a file of that name exists in the working directory. NumPy arrays, PIL images and tensors are converted as before. ([#5231](https://github.com/kornia/kornia/pull/5231))
- `KORNIA_CHECK_IS_IMAGE` range-check fixes. An empty image, such as a `(0, 3, H, W)` batch, passes instead of crashing
  inside `torch.aminmax`, with `raises=False` too. A signed integer image whose `bits` is at least the dtype's width (for
  example `int8` with the default `bits=8` or with `bits=9`, or `int64` with `bits=64`) used to fail for every value,
  because `2 ** bits - 1` was converted to the dtype, where it overflowed. It is now range-checked normally. `uint16`,
  `uint32` and `uint64` images are range-checked instead of crashing. The integer range error states
  `[0, 2 ** bits - 1]` and sets `expected_range` to it, where it used to say `[0, 1]` whatever `bits` was.
  `Image.print` now moves channels-last data to channels-first before rendering it, as `Image.write` does, so an RGB image
  from `Image.from_numpy` prints correctly instead of as a garbled picture. ([#5232](https://github.com/kornia/kornia/pull/5232))
- `otsu_threshold` / `OtsuThreshold` no longer return a threshold of 0 for a `float16` image plane of
  65520 pixels or more (256 x 256 and up): the histogram counts were cast back to the image dtype, so
  their sum overflowed to `inf`, and `bfloat16` counts were rounded to 8 bits. The normalised histogram is
  no longer summed in half precision either, so the thresholds of smaller `float16` and `bfloat16` images
  now match the `float32` call (a 1024-pixel ramp of the levels k / 256 gave 0.4922 in `float16` and
  0.4805 in `bfloat16` against 0.5; all three give 0.5). The counts are now kept in `float32` (`float64`
  for a `float64` image). `normalize_homography`, and the `warp_perspective` and `warp_affine` calls built
  on it, now invert `float16` and `bfloat16` matrices in `float32` and cast the result back: the `float16`
  determinant of a large image's pixel-normalization matrix is subnormal, so a 3000 px `float16`
  normalization was 7 % off and an 11600 px one was NaN, and `bfloat16` on MPS with torch 2.5.1 raised for
  want of a `cross` kernel. ([#5237](https://github.com/kornia/kornia/pull/5237))
- `get_gaussian_discrete_kernel1d` computes the exponentially scaled Bessel terms
  `exp(-sigma**2) I_n(sigma**2)` directly, using float32 intermediates for float16 and bfloat16
  inputs and returning the input dtype. This fixes NaN kernels at large sigma values and, for
  float16 with kernel sizes of at least 5, at ordinary positive sigma values as well.
  Finite float64 outputs also change for sigma values from about 7 to 19: previously inaccurate
  kernels now agree with the normalized reference. For `kernel_size=7, sigma=18`, the maximum
  absolute error per tap decreases from about 0.021 to 4.9e-8. ([#5238](https://github.com/kornia/kornia/pull/5238))
- `ImageModule`, `ImageSequential` and the augmentation containers accept grayscale and other
  non-RGB PIL images (modes `L`, `I;16`, `I;16B`, `I`, `F`, `1`), which used to crash `to_tensor`
  with a `permute` or byte-order error, and convert them to one-channel tensors like a grayscale NumPy
  array; palette images (`P`, `PA`) are converted to RGB, or RGBA with transparency, instead of
  yielding their palette indices. `output_type="pil"` / `to_pil` turn a one-channel tensor into a
  mode `"L"` image instead of raising `TypeError: Cannot handle this data type`, and reject
  multi-channel integer tensors PIL cannot store with a message naming the dtype. `to_pil`, `show()`
  and `save()` clamp a float image to `[0, 1]` and round it to 8 bits on the CPU, the conversion the
  `kornia.models` `save` methods already used: a value above 1 used to wrap modulo 256 (1.1 became
  24), a negative value gave 0 or 231 depending on the method and the torch version, and a `uint8`
  image was multiplied by 255 (200 became 56). A negative value now gives 0 on every torch version and
  a `uint8` image keeps its values. In-range float pixels change by at most one step, from truncation
  to rounding (0.999 now gives 255 instead of 254). `show()` renders bfloat16 outputs, which used to
  raise `TypeError: Got unsupported ScalarType BFloat16`. ([#5239](https://github.com/kornia/kornia/pull/5239))
- `kornia.core.external.LazyLoader` can be copied, deep-copied and pickled (these recursed or failed on the module
  object), and dunder lookups such as `__wrapped__` no longer import, install or ask for a module that is not loaded yet
  (the module metadata `__version__`, `__file__`, `__path__` and `__all__` still loads it). An installed optional module
  now loads when `--doctest-modules` is in `sys.argv`: the `onnxruntime` and `diffusers` loaders used to stay empty in any
  such process, for example a user's own `pytest --doctest-modules` run, and raised `AttributeError: 'NoneType' object has
  no attribute ...`. `ImageModule` and the augmentation containers no longer import PIL through the lazy loader to check
  an argument that is not an image (a keyword such as `scale=0.5` or `data_keys=[...]`), so those calls work on an install
  without the `image` extra. ([#5240](https://github.com/kornia/kornia/pull/5240))
- `torch.compile` can trace a function that builds a `kornia.core.TensorWrapper`: on torch 2.14 and 2.5.1 it used to fail
  inside Dynamo with `RecursionError`, with or without `fullgraph=True`, because `TensorWrapper.__getattr__` looked up the
  wrapper's own unset slots through itself on the instance Dynamo had not initialised yet. The wrapper's own names (its
  slots, `data`, `__dict__` and the copy and pickle hooks) are no longer forwarded to the wrapped tensor, so
  `copy.deepcopy` of a `TensorWrapper`, `Vector2`, `Vector3` or `Scalar` keeps the class instead of returning a plain
  `Tensor`, and a deep-copied `Hyperplane` keeps its `Vector3` normal. Pickling and `torch.save` of a wrapper no longer
  fail after a torch function that pickle cannot store: after `tensor ** w` they raised `PicklingError`, and after
  `torch.unique(w)` `AttributeError`. The copied and pickled state now keeps only the `used_calls` entries pickle can
  store, and a `copy.copy` gets its own `used_attrs` and `used_calls` sets instead of sharing them with the original.
  `TensorWrapper` also gains the operators it lacked: the reflected `2 / w` and `2 // w`; `%`, `**`, `@`, `&`, `|`, `^`,
  `<<` and `>>` in their forward and reflected forms (`w % 2`, `2 % w`); unary `+`, `abs(w)` and `~w`; and `float(w)`,
  `complex(w)` and `operator.index(w)`. The unary operators and the conversions used to raise `TypeError`, and so did the
  binary operators with a Python number or another wrapper as the other operand; with a tensor operand the binary
  operators already worked through `__torch_function__`. An operator returns what the wrapped tensor's operator returns,
  wrapped in the class of the left operand when that is a wrapper and otherwise in the class of the right operand. ([#5243](https://github.com/kornia/kornia/pull/5243))
- `to_onnx` exports the model in eval mode and restores each submodule's training flag afterwards, also when the export
  fails. On torch 2.9 and later a model left in training mode exported an active `Dropout`, so onnxruntime zeroed about
  half of the outputs; the legacy exporter of older torch exported in eval mode but then reset every submodule to the
  model's own mode. The dummy input takes the dtype and device of the model's first floating parameter or, for a model
  without one, of its first floating buffer, so float64, float16 and MPS or CUDA models export with an input of their
  own dtype. The dummy input used to be float32 on the CPU, which failed on an input/weight mismatch or, for a model that
  promotes, exported a float32 input. A dynamic `input_shape` whose rank differs from the pseudo shape now raises
  `ValueError`: a longer one raised `IndexError`, and a shorter one exported, sized silently from the pseudo shape's
  leading entries. A missing output directory raises `FileNotFoundError` before the export runs rather than after it,
  and the export no longer passes through the deprecated `io.BytesIO` destination or, on the dynamo exporter,
  `dynamic_axes`, which warned on every call. `ONNXSequential` and `ONNXModule` now use the `session_options=` they are
  given instead of raising `UnboundLocalError`, and `kornia.onnx.utils.add_metadata` (behind `to_onnx` and the
  `add_metadata` methods) overwrites a key that is already present instead of adding a duplicate that
  `onnx.checker.check_model` rejects. It also merges the duplicate keys of a model tagged by an earlier version into one
  entry with the last value, so such a model passes the check. ([#5244](https://github.com/kornia/kornia/pull/5244))
- `load_state_dict_from_url`, `download_file_from_url` and `download_hf_file` no longer hang on a server that
  accepts the connection and then stops sending: each connection attempt and each read now times out after 30 s
  (`timeout=` overrides it per call; the `KORNIA_DOWNLOAD_TIMEOUT` environment variable, in seconds and read at
  each call, sets the default, which is how callers that take no timeout, such as pretrained model constructors,
  raise it; a number from either that is not greater than 0 and at most `threading.TIMEOUT_MAX`, or a variable
  that holds no number, raises `ValueError` naming its source when a download function is called, and a
  `timeout=` that is a `bool`, a `str` or anything else that is not an `int` or a `float` raises `TypeError`),
  the timeout counts as a transient failure, so it is retried and then hands over to the next source, and the
  call ends with the usual `RuntimeError` naming the source and the cache path. A response body shorter than the
  `Content-Length` that framed it is now a failed transfer too, retried like the timeout, instead of being cached
  as a truncated file that every later call returned; chunked and close-delimited bodies are unaffected. To apply
  both checks these functions now run the transfer themselves instead of calling
  `torch.hub.download_url_to_file`, keeping its temporary-file-then-replace write, hash check and progress bar;
  code that patched `torch.hub.download_url_to_file` to intercept kornia's downloads no longer does so.
  `load_state_dict_from_url` now rejects a `file_name` that is not a bare file name with the `ValueError`
  `download_file_from_url` already raised; an absolute path or a `../` name used to make it overwrite the file
  it pointed at with the checkpoint and then delete the original. Calls that passed a path as `file_name` now
  raise `ValueError` before anything is read or written; a `pathlib.Path` holding a bare name is still accepted.
  Both functions also check their other arguments before touching the cache or the network: a URL whose path
  ends in `/`, `/.` or `/..` raises `ValueError` instead of returning the cache directory or its parent (for a
  list of URLs, a first URL ending in `/` used to be cached under its directory name and now raises too), an
  empty `url` list raises `ValueError` instead of `IndexError`, and a `pathlib.Path` or `bytes` URL, alone or in
  a list, raises a `TypeError` that says what to pass instead of an unrelated `TypeError` or `AttributeError`.
  A keyword `torch.hub.load_state_dict_from_url` does not accept raises `TypeError` at once instead of
  re-downloading the checkpoint and raising `RuntimeError`, and a leading `~` in `model_dir` is expanded instead
  of creating a directory named `~`. `hf_url` percent-encodes the repository and the file name, so `"a#b.pth"`
  fetches that file rather than `a`; a name that was already percent-encoded (`"a%20b.pth"`) is encoded again, so
  pass the plain name (`"a b.pth"`). The retry warnings raised through `download_hf_file` point at the caller's
  line. ([#5247](https://github.com/kornia/kornia/pull/5247))
- `check_safetensors` and `load_safetensors` now raise the documented `ValueError`, naming the file, for several malformed headers that used to escape it. Previously a `dtype` given as a JSON list or object raised `TypeError`, a deeply nested header raised `RecursionError`, and an integer too long for Python to decode, or a shape whose byte size was too large to print, raised a `ValueError` that did not name the file. A shape of many large dimensions is now rejected in linear time, where multiplying it out took about a minute for 100 000 dimensions, and error messages quote header values in shortened form. `check_safetensors` now also rejects any empty tensor shape that torch cannot build, such as `[2**63, 0]` or `[0, 2**62, 2**62]`, so `load_safetensors` can load, on the CPU, every file the check accepts. The header is still not validated beyond what the reader uses: the optional `__metadata__` entry is ignored whatever it holds, and when a key appears twice the last one wins. ([#5248](https://github.com/kornia/kornia/pull/5248))
- `kornia.core.ops.eye_like` and `vec_like` now reject a non-integer `n` with a `TypeCheckError` that names `n` ("n must be an integer. Got: 2.0 (float)"), carrying the offending type as `actual_type` and `int` as `expected_type`. Before, a `float`, `True`, a NumPy float or bool scalar or a float tensor got past the `n > 0` check and failed inside torch with a builtin `TypeError` about the signature of `torch.eye` or `torch.zeros`; a string or `None` failed in that comparison with a `TypeError`; `torch.tensor(True)` and a multi-element tensor failed with a `RuntimeError`; and `False` and `torch.tensor(False)` raised "n must be positive. Got: False". With NumPy 1.x, whose bool scalars still implement `__index__`, `np.True_` was read as 1 and `np.False_` raised "n must be positive"; both are rejected too. `TypeCheckError` is a `BaseError`, and since #5233 also a `TypeError`, so both `except BaseError` and `except TypeError` around such a call catch it. An `int`, a NumPy integer and a 0-d integer tensor are accepted as before, and `n <= 0` raises the same error as before. ([#5251](https://github.com/kornia/kornia/pull/5251))
- `kornia.core._compat.deprecated` no longer turns a decorated class into a plain function: the class keeps `isinstance`, subclassing, class attributes and classmethods, and warns when instantiated. Deprecation messages no longer carry a doubled space before the extra reason. ([#5252](https://github.com/kornia/kornia/pull/5252))
- `guided_blur(..., subsample=s)` and `GuidedBlur(..., subsample=s)` now accept a height or width that is not a multiple of `s`, or is shorter than `s`. They raised a broadcast `RuntimeError` (1-channel guidance) or an `einsum` error (multi-channel guidance) when the height or width was not a multiple of `s`, because the upsampled coefficient maps came out `floor(H / s) * s` pixels tall and wide; an axis shorter than `s` failed earlier, in a zero-sized `interpolate`. The maps are now resized to the input's own size, and an axis shorter than `s` is subsampled to one pixel instead of none. For a size that is not a multiple of `s`, row `i` of the subsampled image is the pixel nearest to `(i + 0.5) * H / rows - s / 2` (a tie rounds up), and likewise for columns: the subsampled grid keeps the alignment with the bilinear upsample it has on a multiple of `s`, so such a size is as close to the `subsample=1` result as a multiple is. Sizes that worked give bit-identical results. The box blur still runs on the subsampled grid, so its padding limit applies there: a call raises the padding `RuntimeError` when the window `(kernel_size - 1) // s + 1` is too wide for that grid to be padded in the chosen `border_type`, as `box_blur` does on it. The default `"reflect"` hits this first, since it cannot pad a one-pixel axis at all, and an axis shorter than `s` is one pixel; `"replicate"` and `"constant"` work at every size. ([#5253](https://github.com/kornia/kornia/pull/5253))
- `bilateral_blur`, `joint_bilateral_blur`, `BilateralBlur` and `JointBilateralBlur` now reject an even (or non-positive) `kernel_size` up front with kornia's "Kernel size must be an odd integer bigger than 0" error; the modules raise it when they are constructed (#5163). Previously such a size was accepted and every call then failed inside the filter: usually with a raw torch shape error (`The size of tensor a (7) must match the size of tensor b (8)`), and on small images, where the unfolded window axis has length 1 and broadcasts, with the odd-size error of `get_gaussian_kernel2d`. Odd kernel sizes are unchanged. ([#5255](https://github.com/kornia/kornia/pull/5255))
- `laplacian`, `Laplacian`, `get_laplacian_kernel2d` and `get_laplacian_kernel1d` now reject a kernel size of 1 (`1`, or `(1, 1)` for the 2D forms) with an error that says why. A 1x1 Laplacian kernel is `[[0.0]]`, so the default `normalized=True` divided by zero and returned an all-NaN image, and `get_laplacian_kernel2d(1)` and `get_laplacian_kernel1d(1)` returned the all-zero kernel. Sizes such as `(1, 3)` and `(3, 1)`, which are a 1-D second difference, still work. `Laplacian` now validates `kernel_size` when it is constructed (odd, positive, the right length, not 1x1), so an invalid size such as `Laplacian(2)` raises there instead of at the first `forward` call. ([#5256](https://github.com/kornia/kornia/pull/5256))
- `load_image` now loads 16-bit grayscale and 16-bit RGBA PNGs as `uint16` with `ImageLoadType.UNCHANGED` instead of raising a decoder error; the 8-bit load types, including the default `RGB32`, now raise a `NotImplementedError` that points at `UNCHANGED` for such files. ([#5258](https://github.com/kornia/kornia/pull/5258))
- `image_to_string`, `print_image` and `Image.print` now print one-channel (grayscale) images instead of raising `RuntimeError` or `IndexError`. For `Image.print` this covers channels-first data; channels-last grayscale images print once #5232 is merged. ([#5259](https://github.com/kornia/kornia/pull/5259))
- `kornia.filters.in_range` and `InRange` now check each Tensor bound on its own. A bound that does not fit the input
  raises a `ValueError` naming it, instead of being accepted whenever the other bound was `(B, C, 1, 1)`. A `(W,)` bound
  used to be applied per column. A bound with the wrong batch, height or width was broadcast into a wrongly shaped mask
  when the image's own dimension was 1, and raised a `RuntimeError` otherwise. A bound with several channels on a
  single-channel image was applied as extra thresholds. A Tensor bound paired with a tuple bound raises a `TypeError`.
  A 1-D bound is read per channel and needs `C` elements (or one), so `(C,)` is accepted. Any other Tensor bound of at
  most four dimensions is aligned with the input `(B, C, H, W)` from its last dimension and each dimension must be 1 or
  the input's. That accepts the documented `(1, C, 1, 1)`. It keeps the shapes that already gave a correct mask next to a
  `(B, C, 1, 1)` bound, such as `(C, 1, 1)`, `(C, H, W)`, `(B, C, H, W)` and a 0-d Tensor, and accepts them for both
  bounds.
  A 1-D bound whose length equals both the channel count and the width used to be applied per column and is now applied
  per channel, without an error.
  The docstring formulas now read `lower ≤ input ≤ upper` with the conjunction over `c = 0 … C − 1`. ([#5260](https://github.com/kornia/kornia/pull/5260))
- Fix `image_to_string`, and with it `print_image` and `Image.print`, printing nothing for very wide images whose resized height rounds down to zero. ([#5264](https://github.com/kornia/kornia/pull/5264))
- `kornia.io.load_image` now raises a `ValueError` that names the file and keeps the decoder's reason, chained from the decoder's exception, when an image cannot be decoded (for example a truncated or corrupt PNG, JPEG or TIFF); it used to raise a bare `Exception` that did not name the file. A missing or unreadable file still raises the same `OSError` as before (`FileNotFoundError`, `PermissionError`). ([#5267](https://github.com/kornia/kornia/pull/5267))
- Edge-aware blur pooling now rejects even edge-dilation kernel sizes with a clear validation error. ([#5268](https://github.com/kornia/kornia/pull/5268))
- Motion blur functions and modules now preserve float64 precision when angles and directions are Python numbers. ([#5270](https://github.com/kornia/kornia/pull/5270))
- In float16, the normalized Pascal kernels no longer overflow. `get_pascal_kernel_2d` returned an all-zero kernel for
  `kernel_size` 9 to 11 and NaN from 12, and `get_pascal_kernel_1d(..., norm=True)` returned zeros from 17 and NaN from
  20, so `blur_pool2d` and `max_blur_pool2d` on float16 input returned zero or NaN images from `kernel_size=9` and
  `edge_aware_blur_pool2d` blended in the same broken blur. Each row is now normalized before it is converted to the
  requested dtype, and the 2-D kernel is the outer product of the normalized rows. float32 and float64 kernels are
  unchanged up to `kernel_size=58`. ([#5271](https://github.com/kornia/kornia/pull/5271))
- `kornia.onnx.download.CachedDownloader` keeps an absolute `cache_dir` absolute and caches only complete downloads, so a transfer that is cut short or interrupted is downloaded again on the next call. It used to store the file under a relative copy of the directory in the current working directory and to keep the partial file forever. A partial file already left in the cache by an earlier version is not detected: delete it once. A model cached under the misplaced relative directory is downloaded again once to the absolute one. ([#5272](https://github.com/kornia/kornia/pull/5272))
- `canny` and `Canny` no longer drop both pixels of an equal-magnitude pair in non-maximum suppression. A step between
  two flat regions puts the edge between two pixels of equal magnitude, and the strict comparison on both sides
  suppressed both, so a clean step gave no edge at all. As in OpenCV's `cv2.Canny`, a pixel now has to be strictly
  greater than its left (upper) neighbour and greater than or equal to its right (lower) one along a horizontal
  (vertical) gradient, so the step keeps its left (upper) pixel and gives a one-pixel-wide edge; diagonal comparisons
  stay strict. Edge maps and the returned magnitude change only at such exact ties, and at pixels of the flat magnitude
  `sqrt(eps)` whose two neighbours along the gradient both lie outside the image: every zero-gradient pixel of an image
  one pixel wide and, where a rounding residue in the gradient sets the direction, pixels of an image one pixel high or
  at an image corner. These were kept with magnitude `sqrt(eps)` and now are not. The neighbours are now compared
  exactly: on float32 the convolution used before could return a rounding residue instead of zero for a tie. The
  thresholds are no longer limited to `(0, 1)`: they are compared with the unnormalised Sobel magnitude, eight times
  what `sobel()` returns by default, which reaches 4 for a unit step in a `[0, 1]` image, so any
  `0 < low_threshold <= high_threshold` is accepted. An input whose channel count is not 1 or 3 raises an `ImageError`
  instead of a raw convolution error. ([#5275](https://github.com/kornia/kornia/pull/5275))
- `VisionTransformer.from_config("vit_s/32", pretrained=True)` now loads its weights. The URL pointed at a Hugging Face repository that does not exist, so the download failed with HTTP 401; the checkpoint is hosted at `kornia/vit_s32_i21k_augreg_i21k_r224`. ([#5277](https://github.com/kornia/kornia/pull/5277))
- `filter3d`, `correlate3d` and `convolve3d` now accept non-contiguous kernels with `normalized=True`, preserving the filtering values and gradients of their contiguous equivalents. ([#5278](https://github.com/kornia/kornia/pull/5278))
- `edge_aware_blur_pool2d` and `EdgeAwareBlurPool2D` now reflect-pad the input by `max(2, kernel_size // 2)` pixels
  before blurring it, so the blur no longer reaches the zero padding of `blur_pool2d` near the image border (#5228).
  The input used to be reflect-padded by a fixed 2 pixels, which covers the blur only up to `kernel_size` 5: an image of
  ones came back with a bottom-right corner of 0.939 for `kernel_size=6`, 0.969 for 7 and 0.931 for 9. A `float16` or
  `bfloat16` input is now blurred in `float32` and cast back. In `float16` the output used to be 0 away from edges from
  `kernel_size` 9 and NaN everywhere from 12, because the normalised Pascal kernel overflowed (#5229). ([#5282](https://github.com/kornia/kornia/pull/5282))
- `EfficientViT.from_config` now loads the pretrained backbone weights: the hosted checkpoints store them under a `backbone.` prefix, which the previous non-strict load silently dropped, so every "pretrained" model was randomly initialised. The load is now strict: a checkpoint whose backbone entries do not match the model raises `RuntimeError` instead of returning a partly loaded model. A state dict saved from the backbone itself, without the prefix, still loads. ([#5283](https://github.com/kornia/kornia/pull/5283))
- `OnnxLightGlue` now goes through the `onnxruntime` lazy loader instead of probing the module with
  `importlib.util.find_spec`, so `kornia_config.lazyloader.installation_mode` applies to it. The probe returned before
  anything reached `kornia.core.external.onnxruntime`, so with the mode set to `"auto"` the constructor raised
  `ImportError` and installed nothing, and with `"ask"` it never asked, while every other `onnxruntime` consumer
  installed `kornia[onnx]` or prompted. Under the default `"raise"` mode a missing `onnxruntime` still raises an
  `ImportError` naming the extra to install, now with the loader's wording: `Optional dependency 'onnxruntime' is not
  installed. Install it with: pip install "kornia[onnx]".` ([#5287](https://github.com/kornia/kornia/pull/5287))
- `Resize` with an integer size, `LongestMaxSize` and `SmallestMaxSize` now return an empty batch with the
  resized spatial shape for a `(0, C, H, W)` input, whatever `p` is, instead of raising `KeyError: 'output_size'`
  (#4429). ([#5288](https://github.com/kornia/kornia/pull/5288))
- `GeometricAugmentationBase2D.inverse_masks` now keeps mask-specific sampling overrides local to the call, so a failed inverse no longer changes interpolation for subsequent images. ([#5300](https://github.com/kornia/kornia/pull/5300))
- `filter2d` and `filter3d` with one kernel per sample now accept an input whose batch and channel axes cannot be merged
  without a copy (#5292). Previously a channels-last input raised torch's "view size is not compatible with input
  tensor's size and stride" when the batch and channel axes were merged for the grouped convolution, and with
  `padding='valid'` so did a `filter2d` input whose batch and channel axes are swapped in memory or that is expanded
  along the batch. A shared kernel and a contiguous input were unaffected, as was `fft_conv`. The error reached
  `gaussian_blur2d` with a per-sample `sigma` on a channels-last image. The output values do not change. ([#5301](https://github.com/kornia/kornia/pull/5301))
- `fft_conv` with `padding='valid'` now raises a `BaseError` for a kernel taller or wider than the input, as `filter2d`
  does (#5285). Previously the kernel was truncated to the input size in the frequency domain and the crop read its
  negative size from the end of the tensor, so on a 5 x 6 image a 7 x 3 kernel returned a 4 x 4 tensor and a 6 x 3
  kernel an empty one. A kernel as large as the input still gives one output pixel, and `padding='same'` is unchanged. ([#5302](https://github.com/kornia/kornia/pull/5302))
- `rgb_to_linear_rgb` no longer returns NaN gradients for inputs below -0.055, and neither do `rgb_to_lab` and
  `rgb_to_luv`, which call it. The forward result is unchanged. ([#5329](https://github.com/kornia/kornia/pull/5329))
- Fix `ZCAWhitening.inverse_transform` to use the fitted sample dimension, restoring data correctly for nonzero and negative `dim` values instead of raising shape errors or mixing feature values. ([#5331](https://github.com/kornia/kornia/pull/5331))
- `ZCAWhitening` now stores its fitted mean and transforms as buffers: `state_dict` saves them, loading a fitted state into an unfitted module fits it, and `.to()` moves and casts them. Before, a fitted module's `state_dict` was empty and `.to()` left the fit on its original device and dtype, so the moved module raised on input of the new dtype. A checkpoint saved before this change still loads with `strict=True` and keeps the current fit. ([#5338](https://github.com/kornia/kornia/pull/5338))
- Fix `apply_colormap` mutating input tensors and rejecting leaf tensors requiring gradients. ([#5341](https://github.com/kornia/kornia/pull/5341))
- Fix `rgb_to_luv` and `luv_to_rgb` returning NaNs for float16 black pixels and their gradients. ([#5342](https://github.com/kornia/kornia/pull/5342))
- `load_state_dict_from_url`, `download_file_from_url` and `download_hf_file` keep handling a failed transfer as such when its temporary `.partial` file cannot be removed, for example because another process still holds it open on Windows. The `PermissionError` from that cleanup used to replace the transfer error: a truncated or timed-out transfer was then not retried, the `RuntimeError` named the `PermissionError` as its cause, and a `KeyboardInterrupt` during the transfer was swallowed and the next source tried. Now the transfer error is kept and the leftover file is reported in a warning, to be deleted by hand. ([#5344](https://github.com/kornia/kornia/pull/5344))
- `add_weighted` and `AddWeighted` now accept tensor `alpha`, `beta` and `gamma` that broadcast to the input shape, such as a 0-d scalar or a per-image `(B, 1, 1, 1)` weight. Before, a tensor coefficient had to match the input shape exactly, even though the docstring said broadcastable shapes were allowed. Shapes that would change the output shape are still rejected. ([#5349](https://github.com/kornia/kornia/pull/5349))
- `MS_SSIMLoss` now computes integer and bool images in the dtype of its Gaussian masks, float32 for a default module
  and float64 for a module moved with `.double()`, and returns the loss in that dtype (#5351). Previously such an image
  reached `F.conv2d` with a dtype that did not match the masks and raised torch's error, `expected scalar type Byte but
  found Float` for uint8. Floating images are unchanged and pixel values are not rescaled. ([#5353](https://github.com/kornia/kornia/pull/5353))
- `rgb_to_hsv` returns hue 0 instead of the period itself for a hue just below 0: `[1, 0, 6e-8]` used to return
  `6.2831855` in float32, above 2π, and `[1, 0, 0.003]` returned `6.28125`, float16's 2π. Hues now stay in `[0, 2π)` in
  every dtype, as documented and as `adjust_hue_raw` returns them. All other values are unchanged. ([#5356](https://github.com/kornia/kornia/pull/5356))
- `get_box_kernel1d` and `get_box_kernel2d` now return an ordinary contiguous tensor, so an in-place edit changes only
  the taps it addresses (#5160). Previously they returned a stride-0 view of one value: writing one tap rewrote every
  tap, and writing more than one raised torch's "more than one element of the written-to tensor refers to a single
  memory location" error. The kernel values and the `box_blur` results are unchanged. ([#5357](https://github.com/kornia/kornia/pull/5357))
- `adjust_hue_raw`, `adjust_gamma` and `solarize` build a Python-float parameter in the image dtype instead of rounding
  it to `float32` first. A `float64` image used to get a `float32`-rounded `factor`, `gamma`, `gain`, `thresholds` or
  `additions` (relative error up to 6e-8, so `solarize(x, 0.1)` left a pixel equal to `0.1` uninverted); it now gets the
  value as given. `float32` results are unchanged, and `float16`/`bfloat16` ones change only where rounding the float
  once instead of twice gives a different value. ([#5358](https://github.com/kornia/kornia/pull/5358))
- `ImageSequential` and `AugmentationSequential` with image and mask data keys compile with `torch.compile(..., fullgraph=True)` on torch 2.5.1: their `isinstance` checks use tuples of classes instead of `X | Y` unions, which that version of Dynamo cannot trace. ([#5361](https://github.com/kornia/kornia/pull/5361))
- `load_image` raises the `OSError` subclass that `open()` raises (`FileNotFoundError`, `IsADirectoryError`, `PermissionError`), naming the path, for a missing, directory or unreadable file of every format. A JPEG path raised a bare `OSError` without the file name, so did a TIFF or WebP directory or unreadable file, and a directory or unreadable file with another suffix raised `ValueError` as an unsupported format. ([#5362](https://github.com/kornia/kornia/pull/5362))
- `MotionBlur` and `MotionBlur3D` now honor `mode="bilinear"` instead of silently using nearest interpolation, and show the selected mode in their representation. `MotionBlur3D` accepts tensor and integer angles and rejects sequences whose length is not three at construction. Nearest-mode outputs and border defaults stay unchanged. ([#5364](https://github.com/kornia/kornia/pull/5364))
- `equalize`, `equalize3d`, `RandomEqualize` and `RandomEqualize3D` now count float16 and bfloat16 histograms in float32. float16 rounded counts above 2048 and their total overflowed to inf once it passed 65504, so a float16 image or volume with about 65500 or more pixels per channel could come back as NaN; bfloat16 rounded counts above 256, which shifted the output: a 300x300 two-level image equalized to 0.992 instead of 1.0. ([#5374](https://github.com/kornia/kornia/pull/5374))
- LightGlue ONNX downloads (`download_onnx_from_url`, which `OnnxLightGlue` uses) now go through `kornia.core.download`. A server that stalls fails the attempt after `KORNIA_DOWNLOAD_TIMEOUT` seconds (30 s by default) instead of hanging, a transfer shorter than its `Content-Length` is no longer cached as the model, transient failures are retried, a `~` in `model_dir` is expanded, and `check_hash=True` also checks a file already in the cache and fetches it again on a mismatch. A failed download now raises `RuntimeError` naming the URL and the cache path, with the original error chained as its cause; it used to raise that error (`urllib.error.HTTPError`, `urllib.error.URLError`) directly. ([#5375](https://github.com/kornia/kornia/pull/5375))
- `MS_SSIMLoss` now filters float16 and bfloat16 images in float32 with autocast disabled and returns the loss in the input dtype, as `ssim` and `SSIMLoss` do (#5148). A float16 pair with local means above about 181, such as pixel-range data at `data_range=255.0`, used to give a NaN loss, and bfloat16 was 2.5 % low on such images (12.06 instead of 12.37). A float16 or bfloat16 module renormalises its rounded Gaussian masks, which no longer summed to one. Under `torch.autocast` a float32 input used to be filtered in the autocast dtype (12.04 instead of 12.37 under CPU bfloat16 autocast, NaN under float16) and is now filtered in float32. Images whose dtype differs from the module's, which used to raise torch's dtype-mismatch error, are now accepted: integer and bool images count as the mask dtype, and the loss is returned in the promoted dtype of the two images. ([#5377](https://github.com/kornia/kornia/pull/5377))
- `rgb_to_hls` returns hue 0 instead of the period itself for a red hue just below 0: `[1, 0, 6e-8]` used to return
  `6.2831855` in float32, above 2π, and `[1, 0, 0.001]` returned `6.28125`, float16's 2π. Hues now stay in `[0, 2π)` in
  every dtype, as documented and as `rgb_to_hsv` returns them. All other values are unchanged. ([#5378](https://github.com/kornia/kornia/pull/5378))
- `filter2d`, `filter2d_separable`, `filter3d`, `fft_conv`, `box_blur`, `gaussian_blur2d`, `laplacian`,
  `get_spatial_gradient_kernel2d`, `get_spatial_gradient_kernel3d`, `spatial_gradient` and `spatial_gradient3d`, and the
  functions and modules that pass these options on, now accept the upper- and mixed-case spellings of `border_type` and
  `mode` that their checks already accepted (#5156). Previously `border_type='REFLECT'` passed kornia's check and then
  raised torch's `NotImplementedError: Unrecognised padding mode REFLECT`, and `mode='Sobel'` or `'Diff'` raised
  `NotImplementedError` from the kernel dispatch. Each spelling now takes the same path and gives the same result as its
  lower-case one, a spelling outside the set still raises kornia's error, and lower-case calls are unchanged. ([#5379](https://github.com/kornia/kornia/pull/5379))
- `Normalize`, `Denormalize` and `Rescale` keep Python-number `mean`, `std` and `factor` arguments as Python numbers and apply them in the input's dtype and device, so a float64 input no longer gets a float32-rounded constant. Such arguments are no longer registered as buffers and `repr` shows them as given; tensor arguments are still non-persistent buffers that move with `.to()`. ([#5380](https://github.com/kornia/kornia/pull/5380))
- `solve_quartic` no longer loses the small real roots of a quartic whose other root pair is much larger. Ferrari's small quadratic factor is now rebuilt from the large one with Vieta's relations instead of a cancelling difference. In float32, `(x - 0.5)(x + 0.25)(x^2 + 1e7)` used to return no real roots and `(x - 0.5)(x + 0.25)(x - 1e4)(x - 2e4)` only the two large ones; both now return all their real roots. In float64, a close small pair next to a large complex pair could lose both roots (`(x - 0.01)(x - 0.0105)(x^2 + 2000 x + 4e9)`) or come back about 1e-7 off; such roots are now accurate to about 1e-15. Rows whose two quadratic factors are not separated (constant terms within a factor of 16) are unchanged. ([#5382](https://github.com/kornia/kornia/pull/5382))
- `grayscale_to_rgb` returns independent storage again instead of an expanded view of its input: writing one channel of
  the result no longer changes the grayscale input and the other two channels, and in-place operations on the result,
  such as `add_`, no longer raise `RuntimeError`. Values and shapes are unchanged; the result is now a copy. ([#5383](https://github.com/kornia/kornia/pull/5383))
- `denormalize` and `Denormalize` check a `(C,)` mean or std against dimension 1, as `normalize` does, so every documented `(B, C, *)` rank works: rank-2 input no longer raises `IndexError`, and rank-3 or rank-5 input whose dimension -3 differs from C, including float statistics, is no longer rejected. ([#5384](https://github.com/kornia/kornia/pull/5384))
- `EfficientViT.from_config` now loads local checkpoint paths as well as checkpoint URLs. ([#5385](https://github.com/kornia/kornia/pull/5385))
- `gaussian_blur2d` now rejects a tensor `sigma` whose batch is neither 1 nor the input batch at its entry with an error
  that names `sigma`, before any kernel is built (#5169). Previously the mismatch surfaced from the kernel batch check of
  `filter2d_separable`, or of `filter2d` with `separable=False`, with a message about the kernel. `bilateral_blur` and
  `joint_bilateral_blur` now raise a kornia error for a float `sigma_color` of 0 instead of Python's `ZeroDivisionError`,
  and `motion_blur`, `motion_blur3d`, `get_motion_kernel2d` and `get_motion_kernel3d` now raise a kornia error for a
  `kernel_size` that is not an `int` instead of a `TypeError`. ([#5387](https://github.com/kornia/kornia/pull/5387))
- `kornia.onnx.download.CachedDownloader`, which `ONNXLoader`, `kornia.models.small_sr` and `kornia.contrib.super_resolution` download through, now uses `kornia.core.download`'s transfer: a server that stalls fails with `TimeoutError` after `KORNIA_DOWNLOAD_TIMEOUT` seconds (30 s when it is unset) instead of blocking forever, and a leading `~` in the cache directory or file path is expanded instead of creating a literal `~` directory. ([#5389](https://github.com/kornia/kornia/pull/5389))
- `AugmentationSequential.transform_matrix` composes the matrix that a nested `AugmentationSequential` recorded for the
  current call. Depending on the child order it used to raise `TypeError`, omit the nested transform, or reuse the
  matrix of the nested container's last direct call. A nested container whose children are all non-rigid is skipped
  like a non-rigid child, in either order, and `ImageSequential.get_transformation_matrix` reads the nested
  container's current matrix too. A plain `ImageSequential` child is still omitted (#4476). ([#5391](https://github.com/kornia/kornia/pull/5391))
- `mean_iou_bbox` now computes float16 and bfloat16 boxes in float32 and returns the IoU in the box dtype. In float16 the IoU came out 0 or NaN once the two areas together overflowed: a 181 x 181 box against itself gave 0 and a 256 x 256 box NaN, so `mean_average_precision` scored a perfect float16 detection of a 512 x 512 box as 0 instead of 1. bfloat16 rounded the areas before the ratio (0.9766 instead of 0.9805 for a 200 x 200 and a 200 x 204 box). float32, float64 and integer boxes are unchanged. ([#5392](https://github.com/kornia/kornia/pull/5392))
- The docstrings of `rotate3d`, `RandomRotation3D`, `RandomAffine3D`, `RandomMotionBlur3D`, their 3D parameter generators and `motion_blur3d` read as if `yaw`, `pitch` and `roll` were Euler angles, most of them as rotations about the x, y and z axes. They now say that the three form one axis-angle vector in degrees, converted with Rodrigues' formula: a single non-zero component is the rotation about its axis, while `(90, 90, 0)` is one turn of 127.3 degrees about `(1, 1, 0) / sqrt(2)`, which neither Euler order gives. `MotionBlur3D` says the same. The behaviour is unchanged. ([#5393](https://github.com/kornia/kornia/pull/5393))
- `Vector3.from_coords`, `Vector2.from_coords` and `Hyperplane.signed_distance` compile with `torch.compile(..., fullgraph=True)` on torch 2.5.1: their `isinstance` checks, and the five others left in `kornia/`, use tuples of classes instead of `X | Y` unions, which that version of Dynamo cannot trace. ([#5394](https://github.com/kornia/kornia/pull/5394))
- `get_gaussian_kernel1d`, `get_gaussian_erf_kernel1d`, `get_gaussian_discrete_kernel1d` and `gaussian` now accept a
  Python `int` `sigma`, and `gaussian` an `int` `mean`, and return the same kernel as for the equal `float`; they
  used to raise a `TypeCheckError` or an `AttributeError`. With no `dtype` the kernel has the default floating-point
  dtype, as for a `float` `sigma`. ([#5396](https://github.com/kornia/kornia/pull/5396))
- A tensor method called through a `TensorWrapper` such as `Vector3`, for example `Vector3(t.clone()).sum()`, compiles with `torch.compile(..., fullgraph=True)`. Dynamo used to fail on it when the wrapper held an intermediate tensor, and on torch 2.5.1 for any wrapper. ([#5397](https://github.com/kornia/kornia/pull/5397))
- `RgbToYuv420`, `RgbToYuv422`, `Yuv420ToRgb` and `Yuv422ToRgb` no longer declare `ONNX_EXPORTABLE = False`, so reading that attribute on them now raises `AttributeError`, as on the other color modules. All four export with `torch.onnx.export(..., dynamo=True)` and match eager execution under onnxruntime. They declare `ONNX_DEFAULT_INPUTSHAPE` like the other color modules; for the two decoders it is the shape of the luma plane. A test now exports every class that declares these attributes with its declared shape and checks that a class keeping `ONNX_EXPORTABLE = False` (`Canny`, #5398) still fails to export. ([#5399](https://github.com/kornia/kornia/pull/5399))
- `KORNIA_CHECK_SAME_DEVICES` now returns `False` with `raises=False` and raises `TypeCheckError` with `raises=True` for an empty list, a non-list container or a list holding a non-tensor. Previously `[]` and tuples returned `True` with `raises=False`, and a non-tensor element raised `AttributeError`, or was compared by its `device` attribute if it had one (a `Vector3`, for example). ([#5401](https://github.com/kornia/kornia/pull/5401))
- `kornia.enhance.normalize`, `denormalize` and the `Normalize`/`Denormalize` modules no longer cast `mean` and `std` to an integer input's dtype. For a uint8 image a fractional statistic truncated to 0, so `normalize(x, 0.5, 0.5)` returned `nan`/`inf`, and the arithmetic wrapped around in the integer dtype: `normalize([0, 128, 255], 100.0, 50.0)` returned `[3.12, 0.56, 3.10]` instead of `[-2.0, 0.56, 3.1]`, and `denormalize([0, 128, 255], 10.0, 2.0)` returned `[10, 10, 8]` instead of `[10, 266, 520]`. An integer or bool input is now converted to torch's default floating dtype first; a bool input, which used to raise `NotImplementedError`, is accepted. Floating and complex inputs are unchanged, and pixel values are not rescaled: give `mean` and `std` in the 0-255 scale for a uint8 image. ([#5407](https://github.com/kornia/kornia/pull/5407))
- `kornia.core.utils._l2_normalize`, the L2 normalisation behind the feature descriptors, no longer
  turns a vector holding a NaN into an all-zero vector. Only a vector whose largest magnitude is
  exactly zero takes the zero branch; a NaN now propagates as it does from
  `torch.nn.functional.normalize`. ([#5409](https://github.com/kornia/kornia/pull/5409))
- `kornia.core.download.load_state_dict_from_url` and `download_file_from_url` now raise `ValueError` for a
  `url` string without a scheme, such as a local path, before looking in the cache. The cache is keyed by the
  URL's base name, so a mistyped local path used to load whichever cached checkpoint shared that name. Load a
  local file with `torch.load`. `ModelBase.load_checkpoint` now expands `~` before checking for a local file.
  (#5410) ([#5410](https://github.com/kornia/kornia/pull/5410))
- Error messages that read "Gotcha <value>" now read "Got <value>", across `kornia.filters`, `kornia.augmentation`,
  `kornia.feature`, `kornia.enhance` and `kornia.constants`, and `filter2d` names a bad `behaviour` as an invalid
  behaviour mode instead of an invalid padding mode. (#5411) ([#5411](https://github.com/kornia/kornia/pull/5411))
- `max_blur_pool2d` and `MaxBlurPool2D` now emit a `DeprecationWarning` when `ceil_mode` is passed. The flag never had
  an effect: it went to a max pool that runs at stride 1, where floor and ceil rounding agree, and the strided blur
  has no rounding mode. Results are unchanged; stop passing `ceil_mode`, which will be removed in a future release. ([#5412](https://github.com/kornia/kornia/pull/5412))
- Geometric annotation replay now uses the transformation parameters supplied to the replay call, even after a later image call updates the augmentation state. ([#5416](https://github.com/kornia/kornia/pull/5416))
- `kornia.metrics.mean_iou_bbox` now computes integer and bool boxes in `float32`, so `uint8`, `int8` and `int16` boxes no longer wrap around in widths, areas and the union (which gave IoUs outside [0, 1] or NaN, accepted inverted `uint8` boxes and rejected wide `int8` ones). The result is still `float32`. An integer box set paired with a float16 or bfloat16 one in `xyxy` or `xywh` format now returns `float32`, as `cxcywh` already did, instead of the half dtype. ([#5417](https://github.com/kornia/kornia/pull/5417))
- `bilateral_blur`, `joint_bilateral_blur`, `BilateralBlur` and `JointBilateralBlur` now reject a tensor `sigma_color` or
  `sigma_space` whose batch is neither 1 nor the input batch at their entry with a kornia error that names the argument
  and both sizes, before any padding or kernel is built (#5430). Previously the mismatch surfaced as a torch broadcast
  error from inside the filter that named neither argument. ([#5432](https://github.com/kornia/kornia/pull/5432))
- `in_range` and `InRange` now respect fractional and out-of-range bounds on integer images without truncating bounds or overflowing the image dtype. ([#5434](https://github.com/kornia/kornia/pull/5434))
- `motion_blur` and `motion_blur3d` (and `MotionBlur`, `MotionBlur3D`) now accept a tensor `angle` or `direction` with a
  Python number for the other again, whatever the tensor's dtype and device, and blur as the two-tensor call does (#5429).
  Previously, since #5270, such a call raised a `TypeCheckError` unless the tensor was in the input's dtype promoted to at
  least float32, and a `DeviceError` for a tensor on an accelerator, because the number was built on the CPU in that
  promoted dtype instead of like its tensor partner. ([#5436](https://github.com/kornia/kornia/pull/5436))
- `total_variation` and `TotalVariation` avoid overflow in adjacent differences for uint8, int8, int16, and int32 images, preserving the output dtypes for both reductions. ([#5439](https://github.com/kornia/kornia/pull/5439))
- `otsu_threshold` and `OtsuThreshold` support `torch.export` on the default path again, and compile with
  `fullgraph=True` without graph breaks: the per-plane histograms are computed in one batched operation, without
  `.item()` or a data-dependent branch for constant planes. ([#5441](https://github.com/kornia/kornia/pull/5441))
- `KORNIA_CHECK_SHAPE` now accepts every tensor shape for `["*"]`, treats an empty pattern as scalar-only, and reports the tensor dimension when a leading-wildcard check fails. ([#5442](https://github.com/kornia/kornia/pull/5442))
- Fixed `dataclass_to_dict` raising `TypeError` for a `namedtuple` reachable from a dataclass, instead of preserving its fields. ([#5444](https://github.com/kornia/kornia/pull/5444))
- The `kornia.core.check` failure paths now report the rejected value: `KORNIA_CHECK_TYPE` handles a `int | str` union instead of crashing while building its message, `KORNIA_CHECK_IS_LIST_OF_TENSOR` names the first non-tensor element, the colour and gray checks print the failing shape and raise their `ImageError` under TorchScript, and `KORNIA_CHECK_DM_DESC` returns `False` (or raises `ShapeError`) for a distance matrix or descriptor of the wrong rank instead of an `IndexError`. ([#5445](https://github.com/kornia/kornia/pull/5445))
- `crop_by_indices` no longer allocates a full-batch input gradient and a copy of the whole output gradient for each image crop when gradients flow to a batch of images, so the backward of slice-mode crops (`RandomCrop`, `RandomResizedCrop`, `CenterCrop2D`) grows linearly with the batch size instead of quadratically. ([#5447](https://github.com/kornia/kornia/pull/5447))
- Named input conversion now binds positional arguments to `forward` parameters in image modules and sequential containers, rather than to the internal PyTorch call wrapper. Selective conversion also handles bound decorated methods and variadic arguments. ([#5448](https://github.com/kornia/kornia/pull/5448))
- Made the eager backward pass of `ColorJiggle` and `ColorJitter` with a fixed `order` about 30x faster (about 170 ms to 6 ms with gradients on a `(8, 3, 64, 64)` CPU batch) by selecting each step with `torch.where` instead of `torch.cond`. Every step is now computed even when its range is neutral, so without gradients a fixed order with a step configured off (for example `hue=0`) costs up to about 1.6x more than before. ([#5450](https://github.com/kornia/kornia/pull/5450))
- `So3.matrix()`, `So3.adjoint()` and `So3 * p` now use the rotation of `q / |q|` for a non-unit quaternion, as `Quaternion.matrix()` does, instead of returning a scaled matrix that is not a rotation and scaling points by `|q|^2`. `Se3` and `NamedPose` inherit the fix. The quaternion is still stored as given, so an `nn.Parameter` stays registered; gradients with respect to it now pass through the normalisation. ([#5452](https://github.com/kornia/kornia/pull/5452))
- `So2` module dtype conversions now preserve both complex rotation components:
  `float()` and `to(float32)` use complex64, while `double()` and `to(float64)` use
  complex128. Parent modules and `Se2` inherit the same behavior; device-only moves
  preserve precision. Complex checkpoint keys and shapes remain unchanged.
  Half precision uses PyTorch's experimental complex32 support. Since PyTorch has no
  complex bfloat16 dtype, bfloat16 casts keep the rotation's existing complex precision
  while following device moves; other floating module state still converts to bfloat16. ([#5453](https://github.com/kornia/kornia/pull/5453))
- `kornia.core.utils.is_autocast_enabled` now reports CPU, CUDA, MPS and XPU autocast on every supported torch version. It
  used to miss MPS autocast on torch 2.5.1 and XPU autocast on every version, so on torch 2.5.1 inside
  `torch.autocast("mps", dtype=torch.float16)` augmentations such as `RandomGaussianBlur` returned float16 for a float32
  input; they now return the input dtype, as they already did on torch 2.14. `LightGlue` now checks autocast for the
  device type of its descriptors, so running it on CPU inside an MPS autocast region no longer raises a dtype-mismatch
  `RuntimeError` on torch 2.14. ([#5454](https://github.com/kornia/kornia/pull/5454))
- `RandomMotionBlur3D` now raises a `ValueError` naming the class, the drawn kernel, the border type, the axis and the input shape when a volume is too small for `border_type="reflect"` or `"circular"` along depth, height or width, as `RandomMotionBlur` does for an image, instead of letting a raw torch padding error out. The sizes it accepts are unchanged. ([#5455](https://github.com/kornia/kornia/pull/5455))
- `kornia.core.utils.batched_forward` now rejects a non-positive `batch_size` instead of failing later with
  `ZeroDivisionError` or an empty-list `torch.cat` error, and `safe_solve_with_mask` / `_torch_solve_cast` reject a
  non-tensor operand with `TypeCheckError` like their siblings `_torch_inverse_cast` and `safe_inverse_with_mask`. (#5456) ([#5456](https://github.com/kornia/kornia/pull/5456))
- On MPS with PyTorch before 2.7, `dilation`, `erosion` and the operators built on them (`opening`, `closing`, `gradient`, `top_hat`, `bottom_hat`) now propagate a NaN inside the window when they use the `shift` engine, which `engine="auto"` (the default) selects on MPS. They used to return a finite value there, because `torch.maximum` and `torch.minimum` ignore NaN on those versions. ([#5457](https://github.com/kornia/kornia/pull/5457))
- `OperationBase` now applies soft `batch_prob` blending independently of the wrapped augmentation's `p` or `p_batch`, so replayed fractional probabilities produce consistent results (#4809). ([#5460](https://github.com/kornia/kornia/pull/5460))
- Fixed `convert_points_to_homogeneous`, and through it `transform_points` and `RANSAC("homography_from_linesegments")`, returning wrong values on MPS once the third-from-last dimension of the points (for `transform_points`, the flattened batch) reaches 65536; RANSAC on line segments picked a wrong homography. ([#5462](https://github.com/kornia/kornia/pull/5462))
- `threshold` and `Threshold` now compare an integer image with `thresh` as given: a fractional threshold is rounded down instead of truncated toward zero, and a threshold outside the image dtype's range passes every pixel or none instead of wrapping or raising. ([#5465](https://github.com/kornia/kornia/pull/5465))
- `RandomCrop(cropping_mode="slice")` now scales bounding boxes and keypoints only along axes where the requested crop exceeds the padded input, matching the image resize. ([#5466](https://github.com/kornia/kornia/pull/5466))
- `RandomMosaic` with `cropping_mode="slice"` no longer offsets bounding boxes from the image content by the
  fractional part of the crop start (up to 1 px up and left): the sampled start corner is now floored to a whole
  pixel, which the image crop and the boxes share. `cropping_mode="resample"` crops from the same integer corner,
  so it no longer interpolates a sub-pixel shift into the image. ([#5467](https://github.com/kornia/kornia/pull/5467))
- `Boxes.compute_area`, `Boxes.transform_boxes` and `Boxes.translate` (and `Boxes3D.transform_boxes`) now accept non-contiguous box tensors, such as a `(B, N, 4, 2)` tensor made by transposing the batch and box axes. They used to raise `RuntimeError: view size is not compatible with input tensor's size and stride`; `compute_area` now returns the documented `(B, N)` areas. ([#5468](https://github.com/kornia/kornia/pull/5468))
- `transform_points` with unbatched `(N, D)` points and a batch of `B > 1` transforms now raises its batch-size `ValueError` also when `N == B`; that case used to fail inside `torch.bmm` with an unrelated `RuntimeError`. `PinholeCamera.project` with `(B, 3)` points on a batch-`B` camera raises the same `ValueError`. ([#5469](https://github.com/kornia/kornia/pull/5469))
- The convolution engine of the morphology operators no longer raises `NotImplementedError` on MPS with torch 2.5.1: the non-finite codes are now matched by scalar comparison instead of `torch.isposinf`/`torch.isneginf`, which lack an MPS kernel there ([#5472](https://github.com/kornia/kornia/pull/5472))
- `solve_quartic` now rescales small-root quartics before evaluating Ferrari's thresholds. It preserves four real roots and rejects non-real candidates when changing the variable's scale previously changed the returned root set. Ordinary and large-root rows retain their existing evaluation path. Rescaling remains bounded by the compute dtype's normal-exponent range and cannot recover underflowed input coefficients. ([#5474](https://github.com/kornia/kornia/pull/5474))
- `otsu_threshold` on the default path now finds the right split for float32 and bfloat16 planes whose range times `nbins` exceeds the float32 maximum (above about 1.3e36 for 256 bins). Such planes used to get a wrong threshold and mask, or a NaN threshold once the range itself overflowed. ([#5479](https://github.com/kornia/kornia/pull/5479))
- Keep binary focal-loss gradients finite for large finite logits with a fractional focusing parameter (`0 < gamma < 1`). ([#5484](https://github.com/kornia/kornia/pull/5484))
- `ScalePyramid` and its feature detector callers can now be constructed under an MPS default device, preserving Gaussian kernel precision and module device placement. ([#5489](https://github.com/kornia/kornia/pull/5489))
- `fit_line` now fits points of dimension 3 and higher at large and small coordinate scales, and in float16 and bfloat16, without overflowing the scatter matrix. A fit with float32 points and float16 weights, or float64 points and float32 weights, no longer raises a dtype error: the line comes back in the promoted dtype, as in 2-D. ([#5498](https://github.com/kornia/kornia/pull/5498))
- `solve_quartic` recomputes float32 rows with a rounding-sensitive resolvent discriminant in float64, preserving close real roots that were misplaced or returned as zero padding. Results retain the input dtype and device; MPS performs the precision retry on CPU with autograd preserved. ([#5501](https://github.com/kornia/kornia/pull/5501))
- Descriptor matching now computes half-precision distances in float32 to avoid overflow and lost nearby matches, while returning distances in the input dtype. ([#5503](https://github.com/kornia/kornia/pull/5503))
- `ImageModule` and `ImageSequential` (core and augmentation) `show()` and `save()` raise the documented `ValueError` before the first call instead of an `AttributeError`, and again after `disable_features = True`, which now clears the cached output instead of leaving the last image to be shown. An output that is not 3-D or 4-D gets one clear `ValueError` from both instead of an empty `ValueError` from `show()` and an `UnboundLocalError` from `save()`. `show()` keeps a unit height or width: a `(3, 1, 5)` image renders as a 5 x 1 RGB picture instead of a 3 x 5 grayscale one. `output_type="numpy"` turns a `bfloat16` output into a `float32` array instead of raising `TypeError`, and the cached output is no longer pickled with the module, so `torch.save(module)` no longer grows by the size of the last output (#5211). ([#5504](https://github.com/kornia/kornia/pull/5504))
- `kornia.metrics.angle_error_mat` and `kornia.metrics.angle_error_vec` now read the sine of the angle from the skew
  part of the relative rotation and from the cross product of the two unit vectors and return `atan2(sin, cos)`,
  so a float32 angle is resolved to about 2e-5 degrees over the whole range (#5500). Previously both used
  `acos` of the cosine, which returned exactly 0 for every angle below about 0.03 degrees and 180 for 179.99
  degrees in float32, and had an infinite or NaN gradient at exactly 0 and 180 degrees on older torch versions;
  the gradient there is now the subgradient 0.
  `angle_error_vec` also scales each vector by its largest absolute entry instead of its norm, so very long and very short
  vectors keep their angle: two float32 vectors 45 degrees apart with lengths 1.8e19, 1.9e19, 1e30, 9e-23 or 1e-40, or two
  float16 vectors with lengths 255, 257 or 6e4, now return 45.0, where previously the squared norms or their product
  overflowed or underflowed and the result was 90, NaN, NaN, 41.4 and NaN in float32 and 90, NaN and NaN in float16. ([#5505](https://github.com/kornia/kornia/pull/5505))
- `ParametrizedLine.intersect` now keeps gradients finite for parallel lines, including mixed batches of parallel and intersecting lines. ([#5517](https://github.com/kornia/kornia/pull/5517))
- `kornia.core.download.load_state_dict_from_url` and `download_file_from_url` no longer advise passing
  `path.as_uri()` when given a `pathlib.Path`: a `file://` URL is cached by its base name like any other URL, so a
  cached checkpoint of the same name, when there was one, was loaded instead of the local file. The `TypeError` now says
  to load a local file with `torch.load`. `DexiNed.load_from_file` in `kornia.filters` and `kornia.models` now reads a
  string that names an existing file with `torch.load(..., weights_only=True)`. (#5477) ([#5520](https://github.com/kornia/kornia/pull/5520))
- `find_essential` now documents that `weights` is intentionally ignored for compatibility, including for `N > 5`, rather than listing it as a known defect. Positive weights leave an exact five-point sample's null space unchanged; for larger sets they can affect the least-squares fit. ([#5522](https://github.com/kornia/kornia/pull/5522))
- `bilateral_blur`, `joint_bilateral_blur`, `BilateralBlur` and `JointBilateralBlur` move a tensor `sigma_space` to the device of a floating input, as they already moved a tensor `sigma_color`. A CPU `sigma_space` with a CUDA or MPS image used to raise a device-mismatch error. ([#5524](https://github.com/kornia/kornia/pull/5524))
- `RandomCrop.inverse` with `padding` and a crop size equal to the input size (#5512) now follows `forward` for a
  hand-made mixed `batch_prob`: the applied rows are inverted and unpadded on their own and the skipped rows stay
  untouched, where previously it raised with the internal padded canvas in the message. A mixed gate whose unpadded
  crop does change the shape is still rejected, and the message now names the unpadded shapes. ([#5525](https://github.com/kornia/kornia/pull/5525))
- `median_blur` now returns NaN for every window of more than one pixel that holds a NaN or an infinity on every backend, as documented. Windows of 7x7 and larger, and inputs that require grad, no longer depend on how the backend's convolution handles `0 * NaN` and `0 * Inf` (CPU bfloat16 on x86 returned finite values there). ([#5529](https://github.com/kornia/kornia/pull/5529))
- `kl_div_loss_2d` and `js_div_loss_2d` now apply the `0 · log 0 = 0` convention, so a distribution with empty cells
  compared with itself gives 0 with finite gradients instead of NaN (#5554). A zero cell also no longer makes a
  gradient NaN where the value was finite: the gradient of `kl_div_loss_2d` with respect to `target`, and of
  `js_div_loss_2d` with respect to either input, was NaN at every zero cell of that input. Every other cell keeps its
  value, including `+inf` where `pred` is zero and `target` is positive, and NaN for a NaN or negative entry and for an
  infinite `pred` where `target` is zero; strictly positive heatmaps are unchanged. ([#5557](https://github.com/kornia/kornia/pull/5557))
- `MILossFromRef`, `NMILossFromRef` and their `2D`/`3D` variants now follow `Module.to(device)` and `Module.to(dtype)`
  (#5547): `bin_centers` is a non-persistent buffer, so a moved module no longer raises a device-mismatch error, and
  `eps` is read from the cached signal's dtype, so a module moved to float64 no longer keeps float32's epsilon. The
  `state_dict` keys are unchanged. ([#5559](https://github.com/kornia/kornia/pull/5559))
- `ssim3d` now builds its Gaussian window in the device and dtype of the inputs, as `ssim` does, and filters mixed
  inputs in their common dtype (#5534). Float64 results are accurate to float64 roundoff instead of carrying the
  rounding of a float32 window, which moved them by under 1e-6 on uniform random volumes and by more on low-contrast
  ones, 8.3e-5 for values in [0.9, 1.0]. Float32 and half results on the CPU are unchanged; on a GPU the window is now
  computed on the device, which moves float32 results by float32 roundoff (1.1e-6 on MPS). ([#5560](https://github.com/kornia/kornia/pull/5560))
- `focal_loss` no longer returns NaN gradients with a fractional focusing parameter (`0 < gamma < 1`) when the softmax probability of any class rounds to 1, for a correct and a wrong prediction alike. ([#5563](https://github.com/kornia/kornia/pull/5563))
- `ssim` now computes a pair of a float64 image and a float32, half-precision or integer image in float64, with the
  Gaussian window built in float64 (#5574). Previously the window was built in the dtype of `img1` and each image was
  filtered in its own dtype, so the float64 result depended on the argument order: on the low-contrast pair of #5574 it
  missed the all-float64 map by 1.1e-4 in one order and 5.7e-5 in the other, and swapping the arguments changed it by
  1.3e-4. Both orders now equal the all-float64 map. Pairs without a float64 image, same-dtype pairs included, are
  unchanged, and `SSIM`, `ssim_loss` and `SSIMLoss` inherit the change. ([#5578](https://github.com/kornia/kornia/pull/5578))
- `mutual_information_loss`, `normalized_mutual_information_loss`, their `_2d` and `_3d` forms and the `MILossFromRef` and
  `NMILossFromRef` modules now give a constant input, target or reference a zero gradient (#5575). Previously the
  discarded branch of the `torch.where` that normalises a constant signal to zero backpropagated `0 / 0`, so the constant
  sample's gradient was NaN, and with it the gradient of any parameter shared across the batch. Loss values are unchanged. ([#5579](https://github.com/kornia/kornia/pull/5579))
- `ImageSequential`, `AugmentationSequential` and `VideoSequential` now accept a nested `PatchSequential` (#5584).
  Previously the parent's shape tracking read the nested container's `PatchParamItem` entries as `ParamItem` and raised
  `AttributeError: 'PatchParamItem' object has no attribute 'data'` for every `PatchSequential` child, with and without
  `patchwise_apply`. The image shape tracked after the nested container is now its input shape, cropped to a multiple of
  the grid size for `padding="valid"`, so a size-dependent sibling such as `RandomCrop` draws its parameters for the image
  it receives. ([#5588](https://github.com/kornia/kornia/pull/5588))
- `charbonnier_loss` and `CharbonnierLoss` now evaluate `sqrt(r² + 1) - 1` as `r² / (sqrt(r² + 1) + 1)` for residuals
  with `r² < 1`, so a small residual keeps its loss of about `r² / 2`. Previously the subtraction cancelled: the loss was
  exactly 0 for `|r|` up to about 4e-4 in float32, 0.04 in float16, 0.1 in bfloat16 and 2e-8 in float64, and inaccurate
  above that (a relative error of 6e-5 at `r = 2**-6` in float32). Losses for `r² >= 1` are unchanged, and the gradients,
  which were already accurate, stay within 2 eps (relative) of their previous values. ([#5589](https://github.com/kornia/kornia/pull/5589))
- `AugmentationSequential` with a `(B, H, W)` mask now compiles with `torch.compile(..., fullgraph=True)`, with and without `keepdim=True`: the mask's batch size comes from the per-sample gate instead of the `forward_input_shape` tensor. A `(B, T, H, W)` mask under `VideoSequential(same_on_frame=True)` no longer raises. ([#5602](https://github.com/kornia/kornia/pull/5602))
- Nested augmentation containers now track size changes correctly during export, and nested `PatchSequential(padding="valid")` containers track their cropped image shape. ([#5603](https://github.com/kornia/kornia/pull/5603))
- `welsch_loss` and `WelschLoss` now evaluate `1 - exp(-r² / 2)` without cancellation for residuals with `r² < 1`.
  Previously the loss was exactly 0 for `|r|` up to about 0.022 in float16, 0.0625 in bfloat16, 2.4e-4 in float32 and
  1e-8 in float64, and lost precision above that. Losses for `r² >= 1` are unchanged, and the gradients, which were
  already accurate, change by about 2 eps (relative) at most. ([#5604](https://github.com/kornia/kornia/pull/5604))
- `normalize_min_max` now returns an empty tensor with the input's shape, dtype and device for an empty batch, and keeps
  the empty leading dimensions of a higher-rank input; it used to raise `ValueError: Invalid input tensor, it is empty.`
  for any empty input. `RandomAutoContrast`, `RandomGaussianIllumination` and `RandomLinearCornerIllumination`, which call
  it, now preserve an empty batch instead of raising that error. Inputs that stay invalid (an empty channel or spatial
  dimension, fewer than two dimensions, a non-tensor) raise the same exception types as before, with new messages. ([#5605](https://github.com/kornia/kornia/pull/5605))
- `charbonnier_loss`, `cauchy_loss`, `geman_mcclure_loss` and their modules evaluate float16 and bfloat16 inputs in
  float32 and return the input dtype. In float16 they returned `inf` or NaN from a residual of 256 (181 for
  Geman-McClure), with a zero, infinite or NaN gradient, and the Geman-McClure gradient was more than 10% off from a
  residual of about 25 in float16 and 10 in bfloat16, and exactly 0 from about 74 and 26. ([#5609](https://github.com/kornia/kornia/pull/5609))
- Indexing a `VideoBoxes` now returns a `VideoBoxes` that keeps `temporal_channel_size`, so `video_boxes[i].to_tensor()`
  no longer raises `AttributeError`. ([#5613](https://github.com/kornia/kornia/pull/5613))
- `AugmentationSequential.inverse` of an empty-batch mask no longer raises `IndexError` (from `padding_size[0]`) when it
  replays `RandomCrop` parameters other than the dict the last forward call used, such as a copy. `VideoSequential` no
  longer raises `IndexError` on an empty batch when it contains a `Resize`, `LongestMaxSize` or `SmallestMaxSize`.
  Part of #4429. ([#5620](https://github.com/kornia/kornia/pull/5620))
- Accept non-contiguous label tensors in `confusion_matrix` and `mean_iou`. ([#5625](https://github.com/kornia/kornia/pull/5625))
- Under graph capture (`torch.jit.trace`, `torch.export`, ONNX export), `safe_inverse_with_mask` and
  `_torch_inverse_cast` now return the finite inverse of a regular 2x2, 3x3 or 4x4 matrix at any scale the inverse fits
  in, and the mask of `safe_inverse_with_mask` is `True` for it (#5507). Previously the closed-form `adj / det`
  overflowed or underflowed the dtype long before the inverse did, so the traced mask rejected `1e13 * I` and
  `1e-13 * I` of order 4 in float32 that eager accepts, and `2**40 * I` came back as the zero matrix with mask `True`.
  The adjugate is now taken on a copy balanced by powers of two per row and column, which leaves every result whose
  adjugate and determinant were normal numbers unchanged to the bit; where one of them overflowed or went subnormal the
  result was `inf`, `nan`, zero or rounded, and is now the inverse. Eager mode is unchanged. ([#5633](https://github.com/kornia/kornia/pull/5633))
- `dice_loss` and `DiceLoss` with `eps=0` no longer return a NaN loss (`'micro'`) or a NaN gradient (`'macro'`) for a sample whose pixels are all ignored, nor a NaN `'micro'` loss for a `weight` that is 0 for every class: such a sample enters the batch mean as loss 1 with a zero gradient, as it does at `eps > 0`. In float16 the `'micro'` gradient for an all-zero `weight` is now finite at the default `eps` too. ([#5637](https://github.com/kornia/kornia/pull/5637))
- `solve_quartic` keeps repeated roots, tells close real pairs from nearby complex pairs across coefficient scales, finds roots that span many decades or sit in a tight cluster, and returns its genuine roots in descending order followed by padding. `solve_cubic` keeps exact double roots. `solve_quadratic` and `solve_cubic` handle coefficients whose products leave the dtype range, float32 cubics are solved in float64 on every device, so a row's roots no longer depend on the rest of its batch, and all three solvers keep finite gradients in mixed batches and support full-graph compilation. On MPS, float32 quadratics and cubics are solved on the CPU in float64, as quartics already were, which fixes flushed subnormal products and misclassified close pairs; this is slower than an on-device solve from about 10^3 rows for quadratics and beyond about 5 * 10^5 rows for cubics. Under `torch.compile` with Inductor on CUDA, Triton's fused multiply-adds can still lose exact double roots of `solve_cubic` and `solve_quartic`; this is documented and not planned to be fixed. ([#5639](https://github.com/kornia/kornia/pull/5639))
- `fit_line` with weights now fits 2-D points at any weight scale, and float16 points whose weighted sums exceed the
  float16 range, as the D ≥ 3 branch already did: each row's weights are divided by their largest value, and float16 and
  bfloat16 fits are computed in float32. Previously, float32 weights of 5e36, and 256 float16 points near x = 1500 with
  unit weights, gave an `inf` or NaN origin and direction, and float32 weights of 1e30 or 1e-30 gave a finite line whose
  direction had a zero gradient with respect to the points. The returned line keeps the dtype promoted from points and
  weights. Other float32 and float64 fits change only by rounding, and are bit-identical when the largest weight of each
  row is a power of two, as with unit weights; float16 and bfloat16 fits can change in the last bits. ([#5641](https://github.com/kornia/kornia/pull/5641))
- `focal_loss` no longer produces NaN losses or gradients from overflowing non-target log probabilities. ([#5642](https://github.com/kornia/kornia/pull/5642))
- `MILossFromRef`, `NMILossFromRef` and their `2D`/`3D` forms store a copy of the `mask` passed at construction. Editing that mask tensor in place afterwards used to change the loss silently (same number of selected positions) or make the forward pass raise `IndexError` (a different number); it now changes nothing. ([#5645](https://github.com/kornia/kornia/pull/5645))
- `keypoints_to_grid`, used by `WunschLineMatcher`, divides by the image size as a tensor. On CUDA it is now correctly rounded and matches CPU and the reference normalization; before, it multiplied by a rounded reciprocal and could differ in the last bit. ([#5646](https://github.com/kornia/kornia/pull/5646))
- `ImageSequential`, `AugmentationSequential` and `VideoSequential` no longer raise `IndexError: index 0 is out of
  bounds for dimension 0 with size 0` on an empty batch when they contain a `CenterCrop`. A stage after the
  `CenterCrop` records the crop size, whatever its `p`, as its forward pass does, so a later `CenterCrop` larger than
  that size raises the same `AssertionError` as for a non-empty batch (#4429). ([#5647](https://github.com/kornia/kornia/pull/5647))
- `kornia.geometry.homography.find_homography_dlt` with the default `solver="lu"` and five or more points, and
  `find_homography_dlt_iterated`, return the homography instead of an all-NaN matrix on exact correspondences
  whose normal matrix has an exactly zero last LU pivot, such as identical point sets, integer shifts and exact
  90-degree rotations of integer keypoints. How often that happens depends on the BLAS and LAPACK in use: 98 of 500
  random sets of 10 integer keypoints with an integer shift on the reporter's Windows machine, 6 of 500 on macOS arm64.
  Only the last pivot being zero means the system has rank 8, so the homography is unique; it now matches
  `solver="svd"` and `cv2.findHomography`. Systems with an earlier zero pivot, such as coincident points, still
  give NaN. ([#5648](https://github.com/kornia/kornia/pull/5648))
- `GaussianBlur2d`, and `UnsharpMask` which blurs with it, now export through the legacy ONNX exporter (`torch.onnx.export(..., dynamo=False)`, the default on torch 2.5.1) with a dynamic batch, height and width. The export used to fail with "Unsupported: ONNX export of convolution for kernel of unknown shape". A per-sample `sigma` keeps a static batch, and its height and width are dynamic too. ([#5650](https://github.com/kornia/kornia/pull/5650))
- `RandAugment`, `AutoAugment` and `TrivialAugment` can now be pickled and saved with `torch.save`. The default magnitude mapping was a local function, so every policy failed to pickle. A restored policy replays the same draw under the same seed. ([#5651](https://github.com/kornia/kornia/pull/5651))
- `gftt_response` returned a NaN gradient wherever the discriminant `trace**2 - 4 * det` of the structure tensor rounded to exactly 0: on every flat 9 x 9 patch of the image (zero padding, a black border, a saturated region) in every dtype, and in float16 and bfloat16 on many textured pixels too (766 float16 and 3364 bfloat16 pixels of a random `(2, 3, 33, 29)` image, out of 5742). The gradient is now finite, and 0 on flat patches. The discriminant is now computed as `(dx2 - dy2)**2 + 4 * dxy**2`, so the response values also move at rounding level, toward the exact smaller eigenvalue: in float32 the largest deviation from it falls from up to 2e-5 to below 2e-7 of the largest response. ([#5653](https://github.com/kornia/kornia/pull/5653))
- `Affine`, `Rotate`, `Scale`, `Translate` and `Shear` now move their constructor tensors with `.to()`. Before, a
  module moved to another device raised in its forward; one moved to another dtype raised for `Rotate`, `Scale` and
  `Affine` without a `center`, and otherwise computed part of its matrix in the old dtype. Plain tensors are stored as
  non-persistent buffers, so `state_dict()` keys are unchanged, and an `nn.Parameter` argument stays a parameter. ([#5663](https://github.com/kornia/kornia/pull/5663))
- Move constructor tensor weights and coefficients with `.to()` in DiceLoss, FocalLoss, BinaryFocalLossWithLogits, LovaszSoftmaxLoss, RgbToRgba, BgrToRgba and AddWeighted, preserving explicit Parameters and checkpoint keys. ([#5664](https://github.com/kornia/kornia/pull/5664))
- `distort_points_kannala_brandt` now returns finite point and camera-parameter gradients at the optical axis, including in mixed batches. ([#5666](https://github.com/kornia/kornia/pull/5666))
- On MPS with torch 2.9.1, or torch 2.5.1 with `PYTORCH_ENABLE_MPS_FALLBACK=1`, `find_fundamental` with
  `method="7POINT"`, or with `method="8POINT"` and exactly eight unweighted points, and the minimal homography and
  fundamental matrix solvers inside `RANSAC` returned wrong models (in our tests RANSAC found no inliers). MPS
  `solve_triangular` misread the strided LU blocks that the shared null-space solver passed to it; the solver now makes
  them contiguous on MPS. Results on the CPU and on MPS with torch 2.14 are bit-identical to before. ([#5667](https://github.com/kornia/kornia/pull/5667))
- Preserve sensor tilt gradients at zero tilt in `distort_points` and `undistort_points`, so gradient-based calibration can learn tilt from a zero initialization. ([#5668](https://github.com/kornia/kornia/pull/5668))
- `filter2d`, `filter2d_separable`, `filter3d`, `fft_conv`, `gaussian_blur2d`, `box_blur`, `laplacian`,
  `bilateral_blur`, `joint_bilateral_blur` and `canny` filter an integer or bool image in float32 with a floating
  kernel, as the color conversions do (#4053). A uint8 image used to blur to zeros, because the fractional kernel taps
  were cast to uint8; integer sums wrapped (a 3x3 box of ones over 100 gave 132); `normalized=True` raised a dtype
  error; a `sigma` below 1 was truncated to 0 and rejected; `bilateral_blur` subtracted uint8 values and blended the
  edges it should keep; and `canny` found no edge on a signed integer image and raised on uint8 or on a 3-channel
  integer image (#5155). Through them, `motion_blur`, `pyrdown` and `ScalePyramid` no longer return zeros for an
  integer image, and `resize` and `rescale` with `antialias=True` no longer reject it. ([#5669](https://github.com/kornia/kornia/pull/5669))
- `ScaleSpaceDetector` now searches exactly `n_levels` response levels per octave and uses the levels below and above them only as NMS neighbours. With a per-level response such as `BlobHessian` or `CornerGFTT` and a `ScalePyramid` with three extra levels, it used to search one more level. That level has the same scale as level 1 of the next octave, so every octave-boundary scale was detected twice, and the near-identical twins failed the ratio test in matching. `HesAffNetHardNet` and `GFTTAffNetHardNet` were affected. Their scale pyramid now also builds one level fewer per octave. ([#5672](https://github.com/kornia/kornia/pull/5672))
- Accept channels-last and other strided heatmaps in convolutional soft-argmax and quadratic subpixel refinement, including strided precomputed NMS masks. Coordinates and responses match contiguous copies. ([#5673](https://github.com/kornia/kornia/pull/5673))
- `extract_tensor_patches` and `combine_tensor_patches`, including their module wrappers,
  now handle empty image batches without ambiguous reshape errors. Patch counts, spatial
  sizes, dtype, device and autograd connections are preserved. `PatchSequential` also
  supports empty batches when its children do, preserving the existing `same` padding
  and `valid` cropping behavior. Geometric patch inverses and mask transforms remain
  unsupported. Part of #4429. ([#5674](https://github.com/kornia/kornia/pull/5674))
- `LightGlue` no longer prints `Loaded LightGlue model` every time it is constructed, and `match_adalam` (also used by
  `GeometryAwareDescriptorMatcher("adalam", ...)`) reports a configuration key it does not recognize with a
  `UserWarning` instead of printing a `WARNING:` line to stdout; the key is still skipped. ([#5676](https://github.com/kornia/kornia/pull/5676))
- `ScaleSpaceDetector` now places each detection where its octave pixel lies in the input image. It used to scale octave coordinates by the nominal spacing `2**o`, but `ScalePyramid` resizes with `align_corners=True`, which maps octave pixel `u` to input pixel `u * (W - 1) / (W_o - 1)` and the same along `y`. Detections from a doubled first octave were up to half a pixel too far from the origin. Detections from every coarser octave were pulled towards the origin, by several pixels on large images. `HesAffNetHardNet`, `GFTTAffNetHardNet` and `SIFTFeatureScaleSpace` (with its default `"patch"` backend) are affected. The `ScalePyramid` documentation now states the map and that `pixel_dists` is nominal. ([#5677](https://github.com/kornia/kornia/pull/5677))
- `extract_patches_simple` and `extract_patches_from_pyramid` now sample the pixels a LAF describes. They built their grids in the corner-aligned frame (`2x / (W - 1) - 1`) and then sampled with `align_corners=False`, so a LAF centred on pixel `x` was read at `x * W / (W - 1) - 0.5`: up to half a pixel off, and stretched by `W / (W - 1)`. An axis-aligned LAF of scale `PS / 2` with odd `PS` now returns exactly the image crop, and a LAF covering a whole `W`-pixel axis has scale `W / 2`, where it used to be `(W - 1) / 2`. Descriptors, LAF orientations and AffNet shapes change for every pipeline that cuts patches from LAFs: `SIFTFeature` and `SIFTFeatureScaleSpace` with the `"patch"` backend, `HesAffNetHardNet`, `GFTTAffNetHardNet`, `KeyNetHardNet` and `KeyNetAffNetHardNet`. The `"pyramid"` SIFT backends are unchanged. ([#5681](https://github.com/kornia/kornia/pull/5681))
- Unweighted 2D `fit_line` now fits float16 and bfloat16 points in float32, as the weighted 2D and D >= 3 paths already did, and returns the origin and direction in the input dtype. In float16 it used to return an infinite origin and a NaN direction when two coordinates differed by more than 65504 (for example points at x = -40000 and x = 40000), and a horizontal or NaN direction when a set of more than 65504 points overflowed the second moments. bfloat16 results change only by rounding. ([#5684](https://github.com/kornia/kornia/pull/5684))
- Weighted `geometry.fit_line` in three or more dimensions no longer allocates a quadratic-size diagonal weight matrix, reducing memory use for large point sets. ([#5685](https://github.com/kornia/kornia/pull/5685))
- `augmentation.CenterCrop`, `augmentation.CenterCrop3D` and `geometry.transform.CenterCrop2D` now raise `TypeError` when `size` is neither an int nor a tuple, and DeDoDe's `ConvRefiner` raises `ValueError` when a depthwise block's `out_dim` is not a multiple of its `in_dim`. Both used to raise a bare `Exception`; code that catches `Exception` is unaffected. ([#5686](https://github.com/kornia/kornia/pull/5686))
- Fix half-precision mutual descriptor matches on MPS in `match_smnn` and replace its quadratic match-intersection allocation with an integer lookup. ([#5687](https://github.com/kornia/kornia/pull/5687))
- `Canny` and `canny` with `hysteresis=True` can now be exported. Under `torch.export` the hysteresis loop is a `torch.while_loop`, so the exported graph repeats until nothing changes for any input, and the dynamo ONNX exporter writes it as an ONNX `Loop` from torch 2.11. Before, both exporters failed on the data-dependent loop. `Canny.ONNX_EXPORTABLE` is now `True`. ([#5689](https://github.com/kornia/kornia/pull/5689))
- `HardNet8(pretrained=False)` now gives distinct, trainable descriptors. Its placeholder PCA projection
  was an all-ones matrix, which mapped every patch to the same descriptor up to sign; it now keeps the first 128 of the
  512 features until the pretrained weights replace it. ([#5693](https://github.com/kornia/kornia/pull/5693))
- `ImageModule`, `ImageSequential` and the augmentation containers keep the container type of a
  multi-output `forward`: a tuple stays a tuple and a one-element tuple stays a one-element tuple
  with the helper features enabled, the same as with `disable_features = True`. A `forward` that
  returns a dict, `None` or a Python scalar passes that output through, with the tensors of a dict
  cached detached, instead of raising `RuntimeError: Unexpected object ...` after the work has
  already run, and a list output converts its elements for `output_type="numpy"` / `"pil"` like a
  tuple does instead of raising `TypeError: Input type not supported`.

  `AugmentationSequential` with `output_type="numpy"` or `"pil"` used to raise
  `TypeError: Input type not supported` for more than one data key or a dict input. It now converts
  the images and masks, positional or in a dict, and leaves a dict value under an unrecognized key
  as it is. A mask converts like the image: a `(B, 1, H, W)` mask becomes a `(B, H, W, 1)` array,
  the layout a NumPy mask input is read in, or one PIL image per sample; a `(B, H, W)` mask returned
  under `keepdim=True` gets its channel axis back first, and a per-sample list of masks converts
  element by element. Boxes, keypoints and labels are not images: with any of them in the data keys
  such a call raises `ValueError` before `forward` runs. ([#5695](https://github.com/kornia/kornia/pull/5695))
- `SIFTDescriptor` now centres its grid of spatial cells on the patch centre `(patch_size - 1) / 2`, where its
  Gaussian weighting window already was. The grid, inherited from pytorch-sift, sat above and to the left of it: by
  1.5 pixels at the default patch size 41, and by more at larger patch sizes. A mirrored or `rot90`-rotated patch
  therefore did not give a permuted descriptor; with the default 8 angular bins it now does, up to rounding. At some
  patch sizes, 41 among them, the centred grid needs a pooling kernel one pixel wider (17 instead of 16 pixels), and
  `get_sift_bin_ksize_stride_pad` returns it with the new `pad`. Descriptors change for `SIFTDescriptor`,
  `sift_describe` and the `"patch"` backend of `SIFTFeature` and `SIFTFeatureScaleSpace`; the `"pyramid"` backends and
  `DenseSIFTDescriptor` are unchanged. A state dict holding the former, untrained pooling kernel still loads. ([#5696](https://github.com/kornia/kornia/pull/5696))
- `MultiResolutionDetector`, the detector of `SIFTFeature`, `KeyNetHardNet` and `KeyNetAffNetHardNet`, no longer reports
  spurious keypoints beside its 15 px border strip. It zeroed the strip before non-maxima suppression, so a pixel next
  to the strip whose real maximum lay inside it survived the suppression. The suppression now reads the whole response
  and the border is applied to its output, as the mask already was, so a keypoint is a maximum of its full `nms_size`
  window wherever it lies. A stronger response inside the strip therefore suppresses a weaker maximum up to
  `nms_size // 2` pixels beside it, as a stronger neighbour does anywhere else, so detections in that band can change.
  `remove_borders` now receives the suppression output instead of the response, and it writes an exact zero, so an
  infinite score inside the strip no longer becomes NaN. ([#5697](https://github.com/kornia/kornia/pull/5697))
- `euler_from_quaternion` now returns a triple that represents the input rotation at gimbal lock (`pitch = ±pi/2`). There, `roll` and `yaw` used to come from `atan2(0, 0)`, so the triple did not match the input and the round trip through `quaternion_from_euler` was off by up to 2 in rotation-matrix entries. It now returns `pitch` exactly `±pi/2`, `roll` exactly `0` and the one determined combination in `yaw`, within `[-pi, pi]`; the input triple itself cannot be recovered at the pole. The gradient there is finite and small, where before it was of order 1e6 to 1e17 (and `nan` on older PyTorch), and rotations with `w = x = 0` (such as `yaw = pi`) keep a finite gradient on PyTorch 2.5.1 as well. Away from the pole nothing changes: outputs are bit-identical. A pitch within `2 * sqrt(eps)` of `±pi/2` is treated as gimbal lock and snapped to the pole. `Quaternion.to_euler` is fixed the same way. ([#5702](https://github.com/kornia/kornia/pull/5702))
- Weighted `geometry.find_homography_lines_dlt` no longer allocates a quadratic-size diagonal weight matrix, reducing memory use for large line-segment sets. ([#5703](https://github.com/kornia/kornia/pull/5703))
- `confusion_matrix`, `mean_iou` and `mean_iou_bbox` now trace under `torch.compile(fullgraph=True)` on torch 2.14: their data-dependent label-range and box checks are skipped wherever `torch.compile` traces them, as they already were under `torch.export`. Eager calls still raise on an out-of-range label or an invalid box; a compiled `mean_iou_bbox` now gives an invalid box an IoU of 0 or NaN instead of raising. ([#5704](https://github.com/kornia/kornia/pull/5704))
- `BlurPool2D` and `MaxBlurPool2D` preserve their stored kernel across forward calls, so earlier input dtypes no longer
  corrupt later floating-point results. The kernel is a non-persistent buffer: module `.to()` moves it while
  `state_dict()` remains unchanged. ([#5707](https://github.com/kornia/kornia/pull/5707))
- `distort_points_kannala_brandt` no longer returns -inf/NaN float16 point gradients for points at a small nonzero normalized radius (below about 4e-3). Its outputs, its camera-parameter gradients and the other dtypes are unchanged. ([#5708](https://github.com/kornia/kornia/pull/5708))
- `canny` and `Canny` with `hysteresis=True` under `torch.jit.trace`, and so the legacy TorchScript ONNX exporter (`torch.onnx.export(..., dynamo=False)`), no longer repeat the hysteresis round only as many times as the traced image needed. The loop is now a scripted function that the trace calls, so the traced graph and the exported ONNX `Loop` run until nothing changes on every input. Before, a longer edge chain than the traced one was cut short without an error, with only a `TracerWarning`. If TorchScript cannot compile the loop, a `RuntimeWarning` reports it. On torch 2.5.1 and 2.9.1, where that exporter is the default, the trace raised `expected padding to be a single integer value or a list of 2 values` instead. ([#5710](https://github.com/kornia/kornia/pull/5710))
- `HausdorffERLoss` and `HausdorffERLoss3D` now trace under `torch.compile(fullgraph=True)` on torch 2.14: their data-dependent label-range check is skipped wherever a graph captures the call, as it already was under `torch.export`. Eager calls still raise on an out-of-range label; a captured call now scores such a voxel in no class and returns a finite loss instead of raising. ([#5712](https://github.com/kornia/kornia/pull/5712))
- Fit planes with reduced SVD factors, avoiding an unused quadratic-size matrix for large point clouds and their backward passes. ([#5718](https://github.com/kornia/kornia/pull/5718))
- `LoFTR("indoor_new")` no longer switches every LoFTR created afterwards in the same process to the `indoor_new` positional encoding. It used to write `temp_bug_fix = True` into the shared module-level `default_cfg` (and into a config dict passed by the caller), so a later `LoFTR("outdoor")` ran with the wrong positional encoding and matched almost nothing. Each `LoFTR` now works on its own deep copy of the config. ([#5720](https://github.com/kornia/kornia/pull/5720))
- `zca_mean`, `zca_whiten` and `ZCAWhitening` fit float16 and bfloat16 data in float32. In float16 the unnormalized
  scatter overflowed, so eight samples of 0 and 255 gave NaN transforms.
  Eager fitting also runs with autocast disabled: under `torch.autocast` a float32 input used to give transforms in the
  autocast dtype (NaN for pixel-range data under float16 autocast), and now gives the float32 transforms it gets outside
  autocast. TorchScript fitting does not disable an enclosing autocast region. ([#5721](https://github.com/kornia/kornia/pull/5721))
- `SOLD2.match` no longer replaces the global `torch.diag` with a tensor, which made every later `torch.diag(...)` call in the process (including `MKDDescriptor(whitening=...)`) raise `TypeError`. `SOLD2(config=cfg)` no longer sets `use_descriptor = True` on the `DetectorCfg` it is given; it works on a copy. `WunschLineMatcher.needleman_wunsch` now allocates its table in the dtype of the scores; float64 scores used to raise an index_put dtype error, and half-precision scores gave a float32 result. `SOLD2.match` now samples each descriptor map with a grid in that map's dtype, so segments in any floating-point dtype, including the float32 segments `SOLD2.forward` returns, match against the descriptors of a float64 or half-precision model; that, and float64 segments with float32 descriptors, used to raise a `grid_sample` dtype error such as `expected scalar type Half but found Float`. ([#5723](https://github.com/kornia/kornia/pull/5723))
- `match_adalam(dm=...)` now uses the distance matrix for the nearest-neighbour matches and their ratio scores; the argument was documented but never read, so any `dm` gave the same matches as no `dm`. `dm` holds plain (not squared) distances, like the `dm` of the other matchers, and AdaLAM squares it internally, so `dm=torch.cdist(desc1, desc2)` gives the matches of `dm=None`, which uses the squared L2 distances as before. A `dm` of the wrong shape raises.

  `match_adalam` with a `config` that does not set `"device"`, and `GeometryAwareDescriptorMatcher("adalam")`, now run on the device of the descriptors; they used to move everything to the CPU and return CPU tensors for CUDA or MPS input. The unique-pair step no longer needs `torch.unique(dim=0)`, which torch 2.5 lacks on MPS.

  `match_adalam(hw1=..., hw2=...)` now honours the image sizes when they are lists; only tuples were used and a list silently fell back to the keypoint spread.

  `get_laf_descriptors` and `LAFDescriptor` changed the caller's descriptor module: they called `.eval()` on it and cast it to the image's device and dtype in place, so a module shared with other code (a train-mode module, a float64 module) was silently switched to eval and to the image's dtype. The descriptor still runs in eval mode with its parameters cast to the image's device and dtype for the call (a TorchScript module runs as a cast copy), and the descriptors are unchanged, but once the call returns the module is as it was: its device, dtype, parameters and the train/eval mode of every submodule. Gradients still reach its parameters. Under `torch.compile` a descriptor whose dtype differs from the image's is now cast as in eager mode, also with `fullgraph=True`; such a compiled call used to raise a dtype error, or, on torch 2.14, break the graph.

  `make_upright` turned the all-zero padding LAF of an unfilled detector slot into a tiny nonzero frame, which `laf_is_filled` reported as a detection; it now returns the padding unchanged, and its gradient at a padding slot is finite. `laf_to_boundary_points` built its circle in float32 for float64 LAFs, so the points carried float32 rounding; it is now built in the LAF's dtype. `laf_from_three_points` now checks its input shape like `laf_to_three_points` instead of failing deep inside the stack or returning a wrong-shaped LAF.

  `HyNet` now raises a `ShapeError` for patches that are not `(B, 1, 32, 32)`, like `HardNet`, `SOSNet` and `TFeat`; a `(2, 1, 40, 40)` input used to return a descriptor of another width, `(2, 1152)`. The `ValueError` for a non-positive `scaling_coef` of `LocalFeature` (and `scale_laf` of `KeyNetHardNet` and `KeyNetAffNetHardNet`) now says `> 0`, matching the check that rejects `0`, and an `int` such as `scale_laf=2` is converted to `float`; it used to pass the constructor and fail in `forward` with a `TypeError`. The keypoint-range error of `LightGlue` and `OnnxLightGlue` used to be an empty `BaseError`; it now names the expected `[-1, 1]` range and the size orders: `image_size` is `(width, height)` for `LightGlue` and `OnnxLightGlue`, `hw1` / `hw2` is `(height, width)` for `LightGlueMatcher`. `gftt_response` now validates `sigmas` like `harris_response` and `hessian_response` (`TypeError` for a non-tensor, `ValueError` unless it has shape `(B,)`); a batch of one with three `sigmas` used to return a `(3, C, H, W)` response. ([#5726](https://github.com/kornia/kornia/pull/5726))
- `LoFTR` misplaced its keypoints for image sizes that are not multiples of 8. It converted the coarse cells of both
  images to pixels with `H0 / ceil(H0 / 8)` instead of the 8-pixel stride of the coarse feature map, cut the
  fine-matching windows of both images on a grid of the wrong stride, and scaled the fine offsets by `H0 / ceil(H0 / 2)`;
  its FPN also upsampled each level to the size of its skip connection, which reads the coarser level at other positions
  than the trained weights expect whenever a side is not 0 or 7 mod 8. On crops of Oxford graf 361 to 366 px high and
  three known homographies, 4 % of the matches were within 1 px of the ground truth, with a median error of 3.1 px and
  errors up to 9.4 px. Coarse and fine cells are now converted with the 8- and 2-pixel strides of the feature maps for
  any image size, `mask0` / `mask1` are read at the pixels the coarse cells sit on, and the FPN upsamples by 2 and crops
  to the skip size: 99 % of the matches are within 1 px on the same crops (median 0.23 px), and on IMC PhotoTourism
  validation with images resized by 0.6 the pose mAA@10 goes from 0.49 to 0.72. Inputs whose heights and widths are all
  multiples of 8 give the same results as before.

  `LoFTR` with `mask0` / `mask1` now runs in float16. It raised a `RuntimeError`, because the similarity -1e9 given to
  masked pairs does not fit in float16, and padded queries overflowed the linear attention's normaliser to inf, which
  made the features nan. Masked pairs now take the lowest finite float16 value, and padded queries, whose output is 0,
  divide by 1 instead of by eps, so they no longer make the output or the gradients nan; float32 results are unchanged. ([#5729](https://github.com/kornia/kornia/pull/5729))
- `DeDoDe.detect` now pads an image to a multiple of 14 the way `DeDoDe.forward` and `DeDoDe.describe` do: after the
  ImageNet normalisation, so the pad is the mean colour. Previously `detect` called directly padded first and normalised
  after, so the network saw a black strip on the bottom and right, and on a 370x504 crop of Oxford graf img1 it selected
  94 of 2000 keypoints different from `forward` (`describe` and `forward` already normalised first). `DeDoDe.describe` now
  also cuts the padding off its descriptor map: with `keypoints=None` it returns the `(B, DIM, H, W)` map its docstring
  states instead of a map padded to a multiple of 14, and the descriptors it samples at the normalised keypoints of
  `detect` come from the right pixels when `H` or `W` is not a multiple of 14. `detect` followed by `describe` now returns
  the same keypoints (in normalised coordinates), scores and descriptors as `forward`, and images whose sides are multiples of 14 are bitwise unchanged. The mean-colour pad
  is the closer one to the reference implementation (Parskatt/DeDoDe), which does not pad. `DeDoDe.forward` with
  `pad_if_not_divisible=False` now runs on the unpadded image; its inner `detect` and `describe` calls used to pad it
  anyway.

  `SOLD2` and `SOLD2_detector` now pick junctions like the reference implementation (cvg/SOLD2). The junction suppression
  keeps the stronger of two junctions that are at most 8 pixels apart along both axes (`grid_size` in general), where it
  kept both when either distance was exactly 8 pixels; and the junctions are in raster order, so the first endpoint of
  every line segment is the one with the smaller row or, on the same row, the smaller column. Before, the junctions came
  in score order. With `use_junction_refinement=False` the endpoints followed the junction score, and
  `[[70, 100], [30, 20]]` came out where the reference returns `[[30, 20], [70, 100]]`. With the default `True` the
  refinement sorted the endpoints, but it breaks ties between candidate positions by endpoint order, so a refined endpoint
  could differ from the reference's (by 0.5 px on a drawn line). With the pretrained weights on Oxford graf img1 the
  junction set drops from 416 to the reference's 399, and in both modes the 246 segments become the reference's 229 with
  identical coordinates.

  `ALIKED.forward_laf` no longer returns mirrored frames. Its 2x2 block is `U · diag(√λ)` of the covariance of the
  soft-argmax weights, and `torch.linalg.eigh` fixes each eigenvector only up to sign and lists them in ascending order of
  eigenvalue, so `det U` was often -1: no keypoint of Oxford graf or boat img1 had a positive determinant with the
  pretrained `aliked-n16`. The first eigenvector is now negated where needed, so every frame has `det > 0`. The ellipse
  `A Aᵀ` and the scale are unchanged; a frame that was mirrored gets a different orientation. The docstring now states
  that the frame's scale is the spread of the soft-argmax weights, about a pixel or less (0.14 to 1.25 px on graf img1),
  not a keypoint size. `compute_affine=False` and `LightGlueMatcher("aliked")`, which reads only keypoint positions and
  descriptors, are unaffected. ([#5731](https://github.com/kornia/kornia/pull/5731))
- Below `p=1`, `RandomEqualize`, `RandomEqualize3D` and `RandomClahe` no longer raise for an out-of-range image the gate skips, and `RandomGamma` no longer gives a skipped sample a NaN gradient where its value is 0. These classes now run the transform on 0.5 for skipped samples; the selected samples and every other augmentation are unchanged. ([#5732](https://github.com/kornia/kornia/pull/5732))
- `laf_is_inside_image` now tests the exact extent of each region. It used to check ten points sampled on the region's
  boundary, so an ellipse that crossed an image edge between two of them, by up to 4.9 % of its extent along that axis,
  was reported as inside. It now compares the exact span of the ellipse along each image axis, `c_i ± ‖A[i, :]‖`, with
  the same `border` and the same closed `0 .. w - 1` / `0 .. h - 1` bounds. `ScaleSpaceDetector` applies the same rule
  to half of each candidate's frame against a 5-pixel octave border, and therefore so do `HesAffNetHardNet`,
  `GFTTAffNetHardNet` and `SIFTFeatureScaleSpace` with its default patch backend: a candidate whose half frame overhung
  that border along x by less than 4.9 % of its radius used to be kept and is now discarded. Its slot goes to the next
  candidate among the `num_features` best of each octave, and stays empty when there is none. On the first image of each
  of the eight Oxford affine sequences, this replaces 10 to 15 of the 16384 detections per pipeline. `SIFTFeature` and
  the KeyNet pipelines are unchanged.

  `LAFAffineShapeEstimator` and `LAFAffNetShapeEstimator` now compose the shape they estimate with the shape of the
  input LAF. Both sample each patch through the upright form of the input LAF, so the estimate is in that frame's
  coordinates; it used to be written over the input shape as if it were measured in the image, which is right only for
  an isotropic input. For an anisotropic or skewed input the output was the correction rather than the corrected shape,
  and applying an estimator to its own output alternated between the region's shape and a circle (axis ratios 2.00,
  1.01, 2.01, 1.02 on a Gaussian blob of ratio 2). The output region is now that frame composed with the estimate, with
  the input's centre and scale and the same `preserve_orientation` behaviour, so an input that already matches the
  region keeps its shape and repeated application converges. An isotropic input gets the same result as before. The
  frames `ScaleSpaceDetector` hands to its affine module are circles in octave pixels, which the octave-to-image factors
  `(W - 1) / (W_o - 1)` and `(H - 1) / (H_o - 1)` stretch by up to 1.8 % between x and y on the Oxford affine images
  (more on very elongated images), so the frames of `HesAffNetHardNet` and `GFTTAffNetHardNet` move by about 1e-4 of
  their scale (median), and on the first images of the eight Oxford affine sequences about 50 of each pipeline's 16384
  descriptors change by more than 0.01. `KeyNetAffNetHardNet` is unchanged.

  `DenseSIFTDescriptor` now spaces its `num_spatial_bins` x `num_spatial_bins` cells `spatial_bin_size` pixels apart, as
  `SIFTDescriptor` and VLFeat's dense SIFT do. They were one pixel apart, so at the defaults the cells of one output
  covered a 7 x 7 window instead of the 16 x 16 pixels of 4 cells of 4-pixel bins, and an edge in that window lit up to
  three cell columns at once. Descriptor values change, and the output size follows (see the breaking change). The
  docstring now states the size rule, the input position of each output and the default `padding` of 1 (it said 0). ([#5734](https://github.com/kornia/kornia/pull/5734))
- `equalize`, `equalize3d`, `posterize` and `sharpness`, and the `RandomEqualize`, `RandomEqualize3D`,
  `RandomPosterize` and `RandomSharpness` augmentations built on them, now return an empty tensor with the input's
  shape, dtype and device for an empty batch, and keep the empty leading dimensions of a higher-rank input. They used to
  raise `ValueError: Invalid input tensor, it is empty.` for any empty input. An empty channel or spatial dimension still
  raises `ValueError`, with a new message. Results for non-empty inputs are unchanged. Part of #4429. ([#5735](https://github.com/kornia/kornia/pull/5735))
- `spatial_softmax2d`, `spatial_expectation2d`, `spatial_soft_argmax2d` and `SpatialSoftArgmax2d` now return an
  empty result for an empty batch or channel axis instead of raising a reshape error. (#5739) ([#5739](https://github.com/kornia/kornia/pull/5739))
- `SIFTFeature` now detects bright blobs as well as dark ones. Its detector, `MultiResolutionDetector`, keeps the maxima
  of its response only, and `SIFTFeature` gave it the single-scale DoG of `BlobDoGSingle`, whose maxima lie at dark
  blobs: a bright blob was never detected, only weaker maxima on the ring around it. `SIFTFeature` now gives the
  detector the magnitude of the DoG, so a keypoint is a maximum or a minimum of the DoG; the DoG detectors of Lowe,
  OpenCV and VLFeat keep both signs too. The suppression rule is not theirs: a keypoint must be the strict maximum of
  the magnitude in its `config["nms_size"]` window (15x15 by default) on its pyramid level, so a stronger response of
  the other sign in that window suppresses a weaker extremum, be it the ring around a blob or a nearby blob of the other
  polarity, where they test each sign against its 3x3x3 scale-space neighbours separately. An image can therefore yield
  fewer extrema in all than it had maxima, and when `num_features` exceeds them, more slots than before are padded with
  a zero response and a zero LAF: the first image of the Oxford graf sequence fills 4509 of the default
  `num_features=8000` slots instead of 5285. `SIFTFeature` returns different keypoints, many of them at bright blobs,
  and its responses are the magnitude of the DoG. An image and its negative give keypoints at the same positions and
  scales, up to rounding and not in half precision; without `upright` their orientations differ by pi.
  `MultiResolutionDetector` itself and the Key.Net features are unchanged. ([#5744](https://github.com/kornia/kornia/pull/5744))
- `DeDoDe.detect` and `DeDoDe.describe` now work under `torch.compile`. They were decorated with
  `@torch.inference_mode()`, and inductor failed on the inference tensors they produce: with
  `RuntimeError: Cannot set version_counter for inference tensor` on torch 2.14, and with the tensor-storage error on
  torch 2.9.1 that the export-support page reported as an inductor bug. They now run under `@torch.no_grad()`, like
  `DeDoDe.forward` compiled before. Gradients stay off and the eager outputs have the same values; they are ordinary
  tensors now instead of inference tensors, so they can be used in autograd or modified in place afterwards. The compiled
  `detect` can return keypoints with equal scores in a different order from eager. ([#5746](https://github.com/kornia/kornia/pull/5746))
- `LoFTR`'s coarse matching no longer returns a pair of masked cells as a match. With `mask0` / `mask1`, a pair whose two cells are both masked got the confidence `1 / (L * S)` from the two softmaxes (a fully masked row or column is uniform there, and all its other entries are exactly 0, so the pair was also a mutual nearest neighbour), and whenever `match_coarse.thr` was below `1 / (L * S)` it was returned with keypoints in the padding of both images. The confidence of a pair that involves a masked cell is now 0 after the softmaxes, and a zero confidence is never a match (a negative `thr` now behaves like `thr=0`), so a masked pair is not returned for any `thr` and `border_rm`. ([#5749](https://github.com/kornia/kornia/pull/5749))
- `warp_perspective` gives the same image for `s * M` as for `M`, and a finite homography gradient (`1/s` times the
  gradient for `M`), when `s` is very large or very small. For a homography with entries between 0.01 and 1,
  `1e15 * M` returned NaN images and `1e-12 * M` NaN gradients in float32, and `1e110 * M` and `1e-110 * M` returned NaN
  images in float64. Under `torch.jit.trace` (float32 and float64) and the legacy ONNX exporter, `1e8 * M` already
  sampled the wrong pixels without any NaN. A matrix whose nonzero entries all have magnitude above 1, or all have
  magnitude below 1, is now divided by a detached scale before the warp, which changes its results at rounding level;
  any other matrix, such as one with `M[2, 2] = 1`, gives bit-identical results. ([#5752](https://github.com/kornia/kornia/pull/5752))
- On CPU, float16 and bfloat16 warps no longer return corrupted values (NaN, values up to 3.4e38, and in bfloat16 a sampling grid off by more than half a pixel) on images of about 128 px and more. `warp_affine`, `warp_perspective`, `remap`, `warp_affine3d`, `warp_perspective3d`, `homography_warp3d`, `homography_warp` with a pixel homography, and the augmentations and crops built on them now build their sampling grid and sample in float32 and return the input dtype; `homography_warp` with a normalized homography, `HomographyWarper` and the depth, elastic and thin-plate-spline warps sample in float32. On torch 2.5.1, which has no CPU half-precision `grid_sample`, these warps now run instead of raising. Other devices and dtypes are unchanged. ([#5758](https://github.com/kornia/kornia/pull/5758))
- `spatial_gradient3d` and `SpatialGradient3d` differentiate an integer or bool volume in float32 and give the result of
  the same volume in float32: uint8 and int8 differences no longer wrap, the mixed second derivatives of an integer
  volume are no longer zero, and a uint8 volume with `order=2` and a bool volume no longer raise.
  `gaussian`, `gaussian_discrete` and the kernel builders that call them (`get_gaussian_kernel1d`,
  `get_gaussian_kernel2d`, `get_gaussian_kernel3d`, `get_gaussian_discrete_kernel1d`) build the kernel of an integer
  `sigma` tensor in the default floating dtype, as `gaussian_discrete_erf` does. `gaussian_discrete` used to cast the
  taps to the sigma's dtype, which made them all 0, and `gaussian` wrapped the offsets of a uint8 sigma, which made the
  taps before the centre wrong. ([#5759](https://github.com/kornia/kornia/pull/5759))
- `DeDoDe.detect` and `DeDoDe.forward` now run with float16 and bfloat16 weights on CPU and MPS. `DeDoDeDetector.forward`
  lifted the decoder context to float32 for the interpolation and fed it back to the half-precision decoder, which raised
  `RuntimeError: Input type (float) and bias type (c10::Half) should be the same`. The context is now converted back to
  the input dtype after the interpolation. Results in float32 and float64 are unchanged. ([#5760](https://github.com/kornia/kornia/pull/5760))
- `SIFTFeature(descriptor_backend="pyramid")` and `SIFTDescriptorFromPyramid` now estimate orientation from smoothed, densely sampled gradient maps without extracting image patches. Descriptor histograms are independent of orientation resolution, and separable spatial pooling reduces CPU extraction cost. ([#5762](https://github.com/kornia/kornia/pull/5762))
- `make_grid`, `ImageModule.show` and `ImageModule.save` no longer raise `ZeroDivisionError` for an empty batch with the
  default `n_row`: `make_grid` returns an empty grid, as it already did for an explicit `n_row`. An `n_row` below 1 now
  raises a `ValueError` naming `n_row` instead of a `ZeroDivisionError` (`n_row=0`) or an unrelated reshape error
  (negative `n_row`). (#5755) ([#5763](https://github.com/kornia/kornia/pull/5763))
- `conv_soft_argmax3d` and `ConvSoftArgmax3d` return correctly shaped empty outputs when the batch or channel axis is empty. ([#5764](https://github.com/kornia/kornia/pull/5764))
- The blur-pool functions and modules export through the legacy ONNX exporter with dynamic batch, height and width. Edge-aware blur pooling combines its edge maps with a boolean OR instead of adding booleans, so it also exports with static shapes, and its dynamo export on torch 2.14 no longer gives a graph that onnxruntime rejects. `EdgeAwareBlurPool2D.forward` takes `epsilon=None` (meaning `1e-6`) by default, so a legacy export of the module no longer adds `epsilon` as a second graph input. ([#5765](https://github.com/kornia/kornia/pull/5765))
- `get_rotation_matrix2d` reports mismatched input dtypes separately from mismatched devices, instead of describing both as a device mismatch. Both checks retain `ValueError` and identify the values for `center`, `angle` and `scale`. ([#5767](https://github.com/kornia/kornia/pull/5767))
- `js_div_loss_2d` evaluates float16 and bfloat16 inputs in float32 before forming the mixture, avoiding infinite loss and non-finite gradients for small positive probabilities while preserving the output dtype. ([#5771](https://github.com/kornia/kornia/pull/5771))
- `cauchy_loss`, `charbonnier_loss`, and `geman_mcclure_loss` promote half-precision inputs before subtracting them, avoiding spurious infinite/NaN losses and NaN gradients when finite inputs have a residual outside the float16 range. ([#5772](https://github.com/kornia/kornia/pull/5772))
- `conv_soft_argmax3d` and `ConvSoftArgmax3d` export with the legacy TorchScript ONNX exporter
  (`torch.onnx.export(..., dynamo=False)`) again; the per-window softmax change of this release (#5134) had broken that
  export with `Unsupported: ONNX export of operator Unfold`. ([#5775](https://github.com/kornia/kornia/pull/5775))
- `welsch_loss` now evaluates float16 and bfloat16 inputs in float32 and returns the input dtype, like `charbonnier_loss`, `cauchy_loss` and `geman_mcclure_loss`: finite float16 inputs whose difference is outside the float16 range no longer give NaN gradients, and half-precision losses are the float32 values rounded once. ([#5777](https://github.com/kornia/kornia/pull/5777))
- `fit_plane` now fits float16 and bfloat16 points in float32 and returns the plane in the input dtype. It used to fit them in their own dtype: a float16 point cloud in millimetres, such as a depth frame, was rejected as collinear because the largest singular value, which grows with the square root of the number of points times their spread, overflowed float16 (#5821), and half-precision offsets lost bits to cancellation (float16 returned -18.5 and bfloat16 -18.0 for an offset of -18.475). The collinearity test keeps the input dtype's tolerance but now applies it to the given points rather than to their half-precision centring, so a near-collinear half-precision set that this rounding let through now raises `ValueCheckError`. float32 and float64 results are unchanged. ([#5783](https://github.com/kornia/kornia/pull/5783))
- `solve_pnp_dlt` with more than six points, and `run_5point` and `find_essential` with more than nine, no longer build the full square matrix of left singular vectors of their linear system, which they never used (2N-by-2N for N points in `solve_pnp_dlt`, N-by-N in the other two). Their memory now grows linearly with N instead of quadratically, with or without autograd, and large point sets run much faster. ([#5786](https://github.com/kornia/kornia/pull/5786))
- Make `AdaptiveDiscriminatorAugmentation(same_on_batch=True)` share its global probability gate across the batch instead of independently selecting image rows. ([#5787](https://github.com/kornia/kornia/pull/5787))
- LoFTR coarse matching now applies either supplied padding mask independently, treating an omitted mask as all-valid. ([#5788](https://github.com/kornia/kornia/pull/5788))
- `rgb_to_hls` and `RgbToHls` now export to ONNX with the TorchScript exporter and compile with
  `torch.compile(fullgraph=True)`. For an input that does not require grad they used to write the H, L and S channels
  with `out=` into views of one preallocated tensor. The TorchScript exporter (the `torch.onnx.export` default before
  torch 2.9, and `dynamo=False` from it) did not record those writes, so the exported graph ignored its input and returned
  zeros, and `torch.compile(fullgraph=True)` raised on the non-contiguous `out=` tensor. The channels are now computed as
  separate tensors and stacked; eager results are bit-identical. ([#5790](https://github.com/kornia/kornia/pull/5790))
- `EfficientViT` now runs with float16 weights on CPU. `LiteMLA.relu_linear_att` lifts a float16 input to float32 for its
  matrix products and returned the float32 result to a float16 projection layer, which raised
  `RuntimeError: expected scalar type Float but found Half`. The result is now converted back to the input dtype.
  float32 and bfloat16 results are unchanged. ([#5793](https://github.com/kornia/kornia/pull/5793))
- `raw_to_rgb` now accepts non-contiguous leading batch dimensions, including video tensors with transposed batch and time axes. ([#5804](https://github.com/kornia/kornia/pull/5804))
- `load_image` returns an 8-bit palette PNG's colours under every load type except `UNCHANGED`, which still returns the
  palette indices: it used to return the indices as gray for every type. `RGBA8` takes alpha from the `tRNS` chunk, and
  the RGB and gray types composite transparent entries over white, as for any RGBA file; under these types, a palette that
  is malformed or too short for its pixels, or a `PLTE` or `tRNS` chunk that is truncated or fails its CRC check, raises
  `ValueError` naming the file instead of being used. 1-, 2- and 4-bit palette PNGs still raise, because kornia-rs cannot
  decode them. A `tRNS` colour key on an 8-bit gray or RGB PNG now makes the key colour transparent in the same way
  instead of being ignored. Compositing a semi-transparent pixel over white rounds to the nearest level instead of
  truncating, which made it up to one level too dark. `write_image` writes 4-channel (RGBA) PNGs and accepts `.tif` and
  extensions in any letter case, such as `.JPG`, all of which it used to reject; a channel count the format cannot store
  raises `ShapeError`, still a `ValueError`, naming the supported counts instead of kornia-rs's "expected 3 channels".
  `get_sample_images` saves a download byte for byte instead of decoding it and re-encoding JPEGs at Pillow's default
  quality 75, so it no longer needs Pillow. It writes each download to a temporary file of its own that replaces the
  cached file only when complete, so an interrupted download leaves no partial file in the cache and concurrent downloads
  of the same image do not interfere. A cache filled by an earlier version keeps its re-encoded files until they are
  deleted. It creates its cache directory only when it has to download, not for local paths, and treats a path as a URL
  only when its scheme is `http` or `https`, so a local file named `http_local.png` loads, and names the cached file after
  the URL's path, without its query or fragment; a URL whose path names no file raises `ValueError` before anything is
  downloaded. ([#5808](https://github.com/kornia/kornia/pull/5808))
- `draw_convex_polygon` overwrites NaN and infinite pixels inside the polygon, which used to come out NaN, and fills a
  `uint8` image with a float colour, which used to raise `RuntimeError`; the colour is cast to the image dtype. Its
  edges are computed in at least float32 and without rounding the slope first, so they add no error to `bfloat16` or
  `float16` vertices, which are used as given (`bfloat16` vertices at x = 260 and 280, both exact, used to fill columns
  259 to 281), a pixel centre lying exactly on a sloped edge is filled, and a long, shallow edge whose vertices lie
  outside the image is placed correctly. `tensor_to_image` returns a `bfloat16` tensor as a `float32` array instead of
  raising `TypeError`. `Image.from_numpy` accepts a 2-D gray array, which raised `IndexError`. `Image.to` and
  `Image.float` set `bit_depth` from the new dtype, so `clone()` of a converted image no longer raises `BaseError`.
  `Image.write` and `Image.print` convert a `BGR` image to RGB first, so the file and the terminal show the true colours
  instead of red and blue swapped. `print_image` accepts any `os.PathLike` path, such as a `pathlib.Path`. Docstrings:
  the `Image.from_dlpack` example passes a `(C, H, W)` array, `Image.height` and `Image.width` name rows and columns the
  right way round, `Image.write` no longer says it writes only JPEG, and the `image_to_tensor`, `tensor_to_image` and
  `draw_convex_polygon` argument descriptions are corrected. ([#5813](https://github.com/kornia/kornia/pull/5813))
- Mutual-information and normalized mutual-information losses now compute half-precision normalization, kernel weights, joint histograms and entropies in float32 and return the input dtype. float16 losses and gradients, NaN from about 184 × 184 pixels with the default kernel, are now finite, and both half dtypes lose the empty-bin bias on small images (12 × 20 seeded pair: float32 −2.662; float16 −2.547 → −2.662; bfloat16 −2.063 → −2.656). The histogram contraction now ignores autocast, so a float32 input inside an autocast region is computed and returned in float32; before, the contraction ran in the autocast dtype, which gave NaN under float16 autocast on large images and a bfloat16 result under CPU bfloat16 autocast. `entropies()` of a half-precision module returns float32. Cached reference modules keep the original dtype's constant-signal threshold and the same state dictionary keys. ([#5815](https://github.com/kornia/kornia/pull/5815))
- `average_quaternions` now compiles with `torch.compile(..., fullgraph=True)`. It used to fail Dynamo graph capture on the data-dependent column index that selects the eigenvector of the largest eigenvalue; it now gathers that column, with the same result in eager mode. ([#5829](https://github.com/kornia/kornia/pull/5829))
- `MKDDescriptor` and `Whitening` with `whitening="pcaws"` and `output_dims` of 40 or less raised `IndexError: index 40 is out of bounds` at construction, because the shrinkage weight (the eigenvalue at index `keval=40`) was read after the spectrum had been truncated to `output_dims`. It is now read from the whitening model's full spectrum, so a smaller `output_dims` keeps the leading columns of the larger transform; `output_dims` of 41 or more gives the same descriptors as before. A `keval` outside `0 <= keval < len(eigvals)` now raises a `ValueError` that says so: a negative `keval` of `Whitening`, which indexed the spectrum truncated to `output_dims` (or raised `IndexError` below `-output_dims`), is no longer accepted.

  `MKDDescriptor` now raises a `ShapeError` (a `ValueError`) for patches that are not `(B, 1, patch_size, patch_size)`; a patch of another size used to fail deep inside with torch's broadcast `RuntimeError`, a three-channel patch with a `ValueError` that reported five channels, and an input that is not 4-D with a plain `ValueError`.

  `laf_is_valid`, `get_laf_scale` and `make_upright` formed products of two LAF entries in float16, which broke every float16 LAF for which such a product exceeds 65504. Any LAF of scale 256 px or more was reported invalid, had an `inf` scale and a `nan` upright frame; `make_upright` already overflowed for an elongated LAF whose first row is 256 px or longer (300 x 20 px, a scale of 77 px), for which `LAFAffineShapeEstimator` returned an `inf` frame. They now compute in float32 for float16 LAFs and return the mask, scale and frame as before; float32, float64 and bfloat16 results are unchanged. With it, `LAFAffineShapeEstimator` and `LAFAffNetShapeEstimator` estimate such a float16 LAF instead of returning a zero-shape or `inf` frame (or the input unchanged with `preserve_orientation=True`), and `SIFTFeatureScaleSpace` with `descriptor_backend="pyramid"` describes it instead of returning the zero descriptor of a padded slot.

  `get_laf_descriptors` and `LAFDescriptor` raised `RuntimeError: view size is not compatible with input tensor's size and stride` when the descriptor module returned a tensor whose per-patch dimensions are not contiguous, such as a transposed block or a channels-last convolution output; the output is now reshaped instead of viewed. ([#5834](https://github.com/kornia/kornia/pull/5834))
- `LightGlue` and `LightGlueMatcher` with point pruning on (the default `width_confidence=0.99`, and the `superpoint`,
  `disk`, `aliked`, `raco-aliked`, `dedodeb`, `dedodeg`, `sift` and `xfeat` feature types) raised `RuntimeError: Index put
  requires the source and destination dtypes match` for float64, float16 and bfloat16 models, and for float32 models with
  `mp=True` on CPU, where autocast computes the scores in bfloat16. This happened at any number of keypoints, because the
  scores are scattered back whenever pruning is enabled. Pruning now keeps the dtype the scores are computed in: the input
  dtype, or with `mp=True` the dtype autocast gives them, bfloat16 on CPU and float32 on CUDA and MPS. The early return
  for every point pruned now gives its zero scores in that dtype as well. The early return for an image without
  descriptors runs no layer and gives the input dtype, so with `mp=True` on CPU its scores are float32 while the other
  paths give bfloat16. Both early returns gave float32 before. Without `mp=True`, float32 results are unchanged. (#5837) ([#5838](https://github.com/kornia/kornia/pull/5838))
- `AugmentationSequential` with a `VideoSequential` child accepts non-contiguous video keypoint and box tensors, clips with no keypoints or boxes, and empty batches, and `VideoKeypoints`/`VideoBoxes` keep those shapes through `from_tensor` and `to_tensor`. Each of these used to raise a `view`/`reshape` `RuntimeError`. ([#5840](https://github.com/kornia/kornia/pull/5840))
- Keep pretrained `SOSNet` and `HyNet` float16 descriptors accurate, called directly or through `get_laf_descriptors` and `LAFDescriptor`, by preserving final BatchNorm statistics and avoiding overflow in filter response normalization. `FilterResponseNorm2d` now computes bfloat16 inputs in float32 too, which brings pretrained `HyNet` bfloat16 descriptors closer to float32. Convert models to float16 before scripting them. ([#5842](https://github.com/kornia/kornia/pull/5842))
- `So2.inverse()`, and so `Se2.inverse()`, now run on CPU float16 rotations, where they raised `NotImplementedError` because the CPU has no ComplexHalf reciprocal: the reciprocal is computed in complex64 and returned in the input dtype. On MPS, float16 inverses of rotations scaled far from unit length no longer return 0 or NaN, and inverses of unit rotations are more accurate. ([#5843](https://github.com/kornia/kornia/pull/5843))
- The README and precision-guide half-precision tables no longer say that `dice_loss` averaging overflows in float16: the overflow was in the float16 reference value of its tests, not in the library. ([#5844](https://github.com/kornia/kornia/pull/5844))
- `AugmentationSequential` annotation-only replays and `inverse` calls now recompute a geometric child's transform matrix when a value in the params dict was replaced or edited in place after the image call, instead of reusing the image call's matrix (#5846). ([#5854](https://github.com/kornia/kornia/pull/5854))
- `AugmentationSequential` called with a dictionary now converts a PIL image or an image file path under an image key (`input`, `image`) as it converts the same value given positionally, instead of failing with `AttributeError`. A path under an image key that does not exist raises `FileNotFoundError`. ([#5855](https://github.com/kornia/kornia/pull/5855))
- `filter2d`, `filter2d_separable` and `filter3d` accept an empty batch with an empty per-sample kernel, so `gaussian_blur2d` with a `(0, 2)` sigma, `motion_blur`, `motion_blur3d`, `RandomGaussianBlur` and fixed-size `RandomMotionBlur` and `RandomMotionBlur3D` return an empty batch instead of raising `RuntimeError: cannot reshape tensor of 0 elements`. Ranged motion-kernel sampling remains outside this fix. Part of #4429. ([#5857](https://github.com/kornia/kornia/pull/5857))
- Palette PNGs with an empty transparency chunk now load with opaque palette entries. ([#5858](https://github.com/kornia/kornia/pull/5858))
- `OnnxLightGlue(device=None)` no longer raises torch's `TypeError` from `torch.device(None)`. `None` is now
  treated as `"cpu"`, as in the other kornia feature constructors, and the `device` docstring says so. ([#5859](https://github.com/kornia/kornia/pull/5859))
- `match_adalam`, `GeometryAwareDescriptorMatcher("adalam")`, `AdalamFilter.match_and_filter` and `AdalamFilter.filter_matches` now return their outputs in the usual order when nothing matches: with fewer than two descriptors on either side, or when no seed survives and no ratio-test match remains, `match_adalam` used to return the `(0, 2)` int64 indices first and the `(0, 1)` float32 distances second, and `AdalamFilter.match_and_filter`, and `AdalamFilter.filter_matches` when no seed survives, returned the float `(0, 1)` tensor as their indices.

  `match_fginn` now masks out the candidates within `spatial_th` of the nearest neighbour when it picks the second neighbour, so the ratio no longer depends on the scale of the distances. It used to add `1e6` to their distance, so with distances of that order, such as squared distances between unnormalised SIFT descriptors, one of them could still be the second neighbour. When every candidate is within `spatial_th`, the match is accepted with ratio `0`; it used to be the nearest distance over the second-nearest plus `1e6`, which is close to `0` only for small distances. ([#5860](https://github.com/kornia/kornia/pull/5860))
- `decompose_essential_matrix_no_svd` no longer returns NaN gradients when the translation is perpendicular to a column of the rotation, for example a stereo rig or forward motion without rotation, or planar motion that rotates about the camera's y axis and translates perpendicular to it. Its outputs are unchanged. ([#5876](https://github.com/kornia/kornia/pull/5876))
- `kornia.contrib.histogram_matching` sums the histogram counts in `int64` and computes the quantiles in
  `float32` (`float64` for a `float64` source) instead of in the input dtype. A `uint8` sum wrapped past 255
  and an `int16` sum past 32767, so a 50x50 `uint8` image mapped to values above 255 and a 64x64 one to NaN
  everywhere; `float16` and `bfloat16` sums stopped being exact past 2048 and 256 pixels, and about half of
  random image pairs at those sizes returned NaN pixels. `kornia.contrib.interp` returns the right endpoint
  on a zero-width last segment instead of NaN, which a template of more than 2**24 pixels could produce in
  `float32`. ([#5878](https://github.com/kornia/kornia/pull/5878))
- `LoFTR.forward` is faster and needs less peak memory: it cuts fine-level windows only at the matched cells, selects
  coarse matches from row and column maxima, runs the coarse-level linear attention as dense matrix products, and
  computes the dual softmax without PyTorch's slow `softmax(dim=1)` kernel outside float16. On 1024 px image pairs from
  the IMC 2021 PhotoTourism validation set (RTX 4090, float32), inference was about 1.3x faster with 30 % less peak GPU
  memory. Outputs change at rounding level, and coarse match selection differs only when a row of the confidence matrix
  has two exactly equal maxima. ([#5879](https://github.com/kornia/kornia/pull/5879))
- `AdaptiveDiscriminatorAugmentation` accepts the dictionary input its signature allows. No dictionary reached
  the augmentations: a plain call raised `ValueError` because the preset always sets `data_keys`,
  `forward(..., data_keys=...)` with the dictionary's own keys raised the same error whenever a sample was drawn,
  and `forward` without `data_keys` raised `KeyError: 'IMAGE'` unless `p` was 0. The dictionary's keys now name
  the data, the same samples are augmented in every entry, and an entry that names no data key is returned
  unchanged. When `p` is above 0, `data_keys` passed with a dictionary must name the same data keys, in any
  order, or a `ValueError` is raised. ([#5882](https://github.com/kornia/kornia/pull/5882))
- ONNX exports of `pyrdown`, `PyrDown`, `build_pyramid` and `rescale` with a dynamic height or width now follow the input size instead of returning the output size of the traced input. Where the exported graph computes the size in float32 (`rescale` under both exporters, `pyrdown` under the dynamo exporter), a factor that is not a power of two can be one pixel off eager (`rescale` 0.7 at a side of 90, `pyrdown` 1.1 at a side of 33); the `rescale` modes that stay static are listed in its docstring. ([#5883](https://github.com/kornia/kornia/pull/5883))
- `AugmentationSequential.inverse` no longer raises `IndexError` when a `VideoSequential` child with `p < 1` skips some frames of a video bounding-box batch: the inverse now lands on exactly the frames that were transformed. `VideoBoxes` index assignment (`video_boxes[key] = other`) no longer raises for clips with no boxes. ([#5884](https://github.com/kornia/kornia/pull/5884))
- `resize`, `Resize`, and `rescale` now support antialiased downsampling of images with a single row or column. ([#5889](https://github.com/kornia/kornia/pull/5889))
- `guided_blur` and `GuidedBlur` convert integer and bool images to float32 before covariance products, avoiding wrapped products and dtype errors while preserving the original image scale. ([#5890](https://github.com/kornia/kornia/pull/5890))
- `total_variation` with `reduction="mean"` now returns finite values for one-row and one-column images, with axes that have no adjacent pixels contributing zero. ([#5891](https://github.com/kornia/kornia/pull/5891))
- `warp_affine3d` raised `ValueError: Expected non-zero, positive output size` for an empty `(0, C, D, H, W)` batch, and so did everything built on it: `affine3d`, `rotate3d`, `crop_by_transform_mat3d`, `RandomAffine3D`, `RandomRotation3D`, `CenterCrop3D` and `RandomCrop3D`. They now return an empty batch of the requested output size, connected to both inputs for backward. ([#5892](https://github.com/kornia/kornia/pull/5892))
- `Rt_to_matrix4x4`, `camtoworld_graphics_to_vision_Rt` and `camtoworld_vision_to_graphics_Rt` raise `ValueError` naming the dtype when `R` and `t` promote to an integer or boolean dtype. They used to raise torch's `RuntimeError: result type Float can't be cast to the desired output type Long` (or `Int`, `Bool`, ...) from inside the 3x4 to 4x4 homography conversion, so code that caught that `RuntimeError` has to catch `ValueError` now. Pairs that promote to a floating point or complex dtype, mixed integer/floating pairs included, return the same matrices as before. ([#5894](https://github.com/kornia/kornia/pull/5894))
- A float32 `LightGlue` with `mp=True` on CPU now gives bfloat16 zero scores when point pruning removes every point, for
  every feature type, like the scores of the same model when it keeps points. With `input_dim` equal to `descriptor_dim`
  (the `superpoint`, `dedodeb` and `dedodeg` feature types, and `LightGlue(None)` by default), it gave float32
  `matching_scores0`, `matching_scores1` and `scores` there before, and `LightGlueMatcher` returned its empty scores in
  float32. (#5877) ([#5896](https://github.com/kornia/kornia/pull/5896))
- A misspelt name given to the `.get` of a `kornia.constants` enum (`Resample`, `BorderType`, `SamplePadding`, `DType`,
  `DataKey`), and an unsupported `torch.dtype` given to `DType.get`, now raise an error that names the value and lists the
  valid names. This covers the `kornia.augmentation` arguments that are looked up in these enums, such as `resample` and
  `padding_mode` of `RandomAffine`, `border_type` of `RandomGaussianBlur` and `data_keys` of `AugmentationSequential`,
  and the `format` of `kornia.contrib.object_detection.results_from_detections`. The error is a `ValueError`, like the
  one for an out-of-range integer, and still a `KeyError`. A value of the wrong type raises a `TypeError` that names the
  enum and the value, instead of listing every member of both enums. ([#5908](https://github.com/kornia/kornia/pull/5908))
- `kornia.filters.guided_blur` and `GuidedBlur` accept an empty batch with a multichannel guidance. They raised
  `RuntimeError: cannot reshape tensor of 0 elements` because the input channel count was inferred with `-1`, which
  an empty tensor cannot provide; this happened both when the guidance is the input and when it is a separate tensor. ([#5909](https://github.com/kornia/kornia/pull/5909))
- `kornia.core.utils.dict_to_dataclass` now converts nested dicts in fields annotated `Optional[...]`, other unions,
  `list`, `tuple` and `dict` of a dataclass, and in string annotations (including modules with
  `from __future__ import annotations`), and keeps a value that is already an instance of the field's dataclass
  instead of raising `TypeCheckError`. The API reference now renders the `kornia.core.utils` device and autocast
  helpers that the `kornia.utils` deprecation warnings point to. ([#5910](https://github.com/kornia/kornia/pull/5910))
- `kornia.onnx.utils.ONNXLoader` caches every remote file under a path of its own. `HFONNXComunnityModelLoader.load_preprocessing`, which `DepthAnythingBuilder.build` calls, caches each repository's `preprocessor_config.json` beside its model in `<cache_dir>/<model_name>/`; every repository used to share `<hub_cache_dir>/preprocessor_config.json`, so a model built after another one could load the other model's preprocessing. `ONNXLoader.load_model("hf://<folder>/<a>/<b>")` caches `<a>/<b>.onnx` instead of `<a>.onnx` for every model under `<a>`. `load_model` and `load_config` cache a URL at `<scheme>/<host>/<directories of its path>/<file name>` under `cache_dir` or the default directory, with `%3A<port>` after the host when the URL has a port and a `%3F<hash>` directory before the file name when it has a query. A URL used to be cached under its last path component alone, query included, so two URLs that ended in the same file name, such as the `onnx/model.onnx` of two onnx-community repositories, shared one file. Files cached under the old names are neither reused nor moved, since they may hold another file's content: the affected configs and models are downloaded once more, and the old files, `<hub_cache_dir>/preprocessor_config.json` among them, can be deleted. The ONNX models of `HFONNXComunnityModelLoader` and `DepthAnythingBuilder`, and `hf://<folder>/<name>` models, keep their cache paths. `kornia.onnx.download.CachedDownloader.download` and `download_to_cache` accept a path without a directory part, which is a file in the working directory (or `cache_dir=""`), instead of raising `FileNotFoundError`. `kornia.io.get_sample_images` takes its download timeout from `KORNIA_DOWNLOAD_TIMEOUT` (30 s when it is unset), as kornia's other downloads do, instead of always using 30 s. ([#5911](https://github.com/kornia/kornia/pull/5911))
- The deprecated `kornia.utils` names and the deprecated top-level aliases such as `kornia.create_meshgrid` work as the
  0.8.2 functions did again. `kornia.utils.<name>` resolves after a plain `import kornia`, where it raised
  `AttributeError` unless `kornia.utils` had been imported first, and the 0.8.2 path
  `kornia.utils.sample.get_sample_images` is back as a deprecated alias. Each shim takes the signature of the function it
  forwards to, so `inspect.signature` and `help()` show the real parameters. Code that calls a shim compiles with
  `torch.compile(fullgraph=True)` and scripts with `torch.jit.script` wherever the function it forwards to does: the
  deprecation warning is skipped while a graph is captured and inside TorchScript, and eager calls still warn.
  `kornia.losses.one_hot` scripts with `torch.jit.script`.
  `torch.load(..., map_location=kornia.utils.map_location_to_cpu)` warns at the line that called `torch.load`, so the
  default warning filters show it there once. `from kornia import to_jax`, and the same for `to_numpy`,
  `to_tensorflow` and `kornia.transpiler`, imports again, and calling the name raises the `RuntimeError` that explains
  the removal instead of a bare `ImportError`. The deprecation messages no longer say that the `kornia.utils` module has
  been removed. ([#5912](https://github.com/kornia/kornia/pull/5912))
- `KMeans.cluster_assignments` now uses the final cluster centers, matching `predict` on the training data when fitting stops before assignments have converged. ([#5922](https://github.com/kornia/kornia/pull/5922))
- `RandomMotionBlur` and `RandomMotionBlur3D` with a ranged `kernel_size` accept an empty batch and return it unchanged.
  They raised before the blur ran because they choose the batch's kernel size from the sampled sizes, of which an empty
  batch has none: `RandomMotionBlur` with `IndexError: list index out of range` and `RandomMotionBlur3D` with
  `RuntimeError: a Tensor with 0 elements cannot be converted to Scalar`. ([#5923](https://github.com/kornia/kornia/pull/5923))
- `SOLD2.match` now returns an int64 tensor when either set of segments is empty (it returned int32), and `XFeat.detectAndComputeDense` with `multiscale=True` no longer returns every cell of a scale whose share of a small positive `top_k` rounds down to 0: the budget is split `int(0.2 * top_k)` / the rest, so for a positive `top_k` the result never exceeds it (zero and negative values still keep every cell) (a default `top_k=4096` now yields 4096 keypoints instead of 4095 on images with enough cells). ([#5926](https://github.com/kornia/kornia/pull/5926))
- `LightGlue` matches a batch of image pairs with point pruning, which is on by default, instead of raising an
  `IndexError`: each pair gives the matches and scores it gives on its own, and a batch that can prune runs its pairs one
  at a time. A batch with too few keypoints for pruning to start (on CUDA, 1536 by default) no longer raises either, and
  early stopping ends such a batch once every pair meets its own criterion, rather than counting the unconfident points
  of all pairs together. A float16 model with `mp=True` on CPU now autocasts to float16 and returns float16 scores,
  instead of raising a `RuntimeError` in its first layer, or with `"keynet_affnet_hardnet"` already when it reads the
  LAFs. ([#5928](https://github.com/kornia/kornia/pull/5928))
- `So3.matrix()` (and `adjoint()`), `So3 * So3`, `So3 * points`, `Se3 * Se3` and `Se3 * points` now compute in float32
  for float16 and bfloat16 inputs and return the input dtype, rounding once; `Se3.matrix()` and `Se3.inverse()` follow
  through `So3`. Evaluated in half precision, each drifted 3 to 6 units of roundoff from the exact result, and
  identities such as `R.T @ R == I`, `(s * s.inverse()).t == 0` and chained point transforms missed by up to about 12
  units; they now hold to about 2. float32 and float64 results are unchanged. Building a bfloat16 `So2` or `Se2`
  (`exp`, `identity`, `random`, `from_matrix`, `Se2.trans`, `trans_x`, `trans_y`, and a `NamedPose` of an `Se2` pose)
  used to fail inside `torch.complex` with a `RuntimeError`, since PyTorch has no complex bfloat16 dtype; it now raises
  `TypeCheckError` naming the limitation. `.bfloat16()` on a built `So2` or `Se2` keeps working and keeps the
  rotation's complex precision. ([#5936](https://github.com/kornia/kornia/pull/5936))
- `kornia.color.apply_colormap` and `kornia.color.ApplyColorMap` accept empty tensors. An empty batch or an empty
  channel dimension raised `RuntimeError: cannot reshape tensor of 0 elements`, and a floating image with zero height
  or width raised `IndexError` from `amax`. They now return an empty tensor of the documented `(B, 3*C, H, W)` shape.
  Non-empty outputs are unchanged. ([#5937](https://github.com/kornia/kornia/pull/5937))
- `DeDoDe()` with the default `amp_dtype=torch.float16` now runs `describe` and `forward` on CPU and MPS with float32 weights and images instead of raising `Input type (c10::Half) and bias type (float) should be the same`. `amp_dtype` is CUDA mixed precision in every component: the `G` descriptor's DINOv2 backbone is cast to it only for CUDA inputs, and on other devices follows the dtype of the input (a float32 backbone stays float32, a `.half()` model gets a half one), so a `.half()` or `.bfloat16()` model with a different `amp_dtype` also runs there now. The CUDA path casts on the host and then moves the backbone, as before; the one difference is that the cast happens at the first CUDA call instead of at construction, so about 0.6 GB more host memory is held in between. ([#5939](https://github.com/kornia/kornia/pull/5939))
- AdaLAM now retains well-supported exact local affine correspondences, including self-matches and noise-free translations, while discounting repeated keypoints and the points used to fit each hypothesis. ([#5941](https://github.com/kornia/kornia/pull/5941))
- `ImageStitcher.postprocess` now removes only the empty columns at the right edge of the mask, and keeps every column that has a covered pixel in any sample. It used to cut at the first empty column. A full-height gap in the mask, such as an invalid column in a `stitch_pair` mask or a pair of images that do not overlap, dropped every column after the gap. A first column with no covered pixel made it return the full width, empty columns included. Masks whose covered columns start at the first column and have no gap, such as those `forward` builds for overlapping images, give the same result as before. ([#5942](https://github.com/kornia/kornia/pull/5942))
- `ObjectDetector.visualize` and `ObjectDetector.save` accept supplied detection tensors, including tensors with no detections, without raising an ambiguous truth-value error. ([#5945](https://github.com/kornia/kornia/pull/5945))
- `unsharp_mask` and `UnsharpMask` filter an integer or bool image in float32 and return float32 on its own scale, like `gaussian_blur2d`, instead of raising a dtype error. ([#5946](https://github.com/kornia/kornia/pull/5946))
- `render_gaussian2d` evaluates float16 `mean` and `std` in float32 and returns the heatmap in float16. A narrow float16 Gaussian, such as a one-pixel `std` in normalized coordinates on a grid more than about 550 pixels wide, used to return a NaN heatmap and NaN gradients. ([#5948](https://github.com/kornia/kornia/pull/5948))
- `warp_grid`, `warp_grid3d`, `homography_warp` (with the default `normalized_homography=True`), `HomographyWarper`,
  `homography_warp3d` and `warp_perspective3d` now give the same result for `s * H` as for `H`, apart from grid points
  whose homogeneous coordinate is still within `1e-8` of zero after the matrix is rescaled. A homography and any
  nonzero multiple of it describe the same projective map, but the grid passed every point with `|w| <= 1e-8` to
  `grid_sample` undivided, so the arbitrary scale of the matrix changed the output without a NaN or an error: for a
  homography with entries between 0.01 and 1, `1e-8 * H` (`1e8 * M` for `warp_perspective3d`, which inverts its pixel
  matrix first) differed from `H` by up to 0.50 of the image range in float64. The grid warps now divide the matrix by
  the detached projective scale #5752 introduced for `warp_perspective`, so gradients follow `dL/d(s*H) = (dL/dH)/s`.
  A matrix whose nonzero entries all have magnitude above 1, or all below 1, changes its results at rounding level; any
  other matrix, such as one with `H[2, 2] = 1`, gives bit-identical results. `warp_perspective3d` rescales the
  normalized inverse of its pixel matrix, so its results can change at rounding level even when `M[3, 3] = 1`. ([#5949](https://github.com/kornia/kornia/pull/5949))
- `kornia.geometry.bbox.transform_bbox` accepts zero boxes and an empty batch. A `(0, 4)` box tensor, or a
  `(0, N, 4)` or `(0, N, 4, 2)` batch, raised `RuntimeError: cannot reshape tensor of 0 elements` and now returns an
  empty tensor of the input's shape. Non-empty results are unchanged. ([#5950](https://github.com/kornia/kornia/pull/5950))
- `combine_tensor_patches` and `extract_tensor_patches` now recombine and extract patches exactly:

  - An all-zero padding or unpadding tuple such as `(0, 0)` or `(0, 0, 0, 0)` no longer switches off `allow_auto_padding` / `allow_auto_unpadding`: a 5 x 7 image with 2 x 2 windows gave 6 patches and dropped the last row and column, and now gives the same 12 patches as `padding=0`.
  - `combine_tensor_patches` divides each covered pixel by its exact overlap count instead of `count + eps`, so a `float64` value of `1000.0` is no longer returned as `999.99999`. Pixels no patch covers are `0`, where `eps=0` gave `nan` at them and a negative `eps` gave `inf` at singly covered pixels.
  - `combine_tensor_patches` recombines integer patches in `int64` without overflow, so `16777217` is no longer returned as `16777216` and `2**63 - 2` no longer as `2**63 - 1`: the overlap mean is exact for every integer value and truncates toward zero. On MPS this needs torch 2.9 or newer: older torch keeps the float32 sum, exact up to 2**24. Complex patches are folded as complex instead of losing their imaginary part.
  - `combine_tensor_patches` sums `float16` and `bfloat16` patches in `float32`: 289 overlapping values up to 255 overflowed to `inf` in `float16` and were off by 64 in `bfloat16`.
  - The `eps` argument of `combine_tensor_patches` is deprecated and has no effect; a value other than the default used to change every recombined pixel. Passing it emits a `DeprecationWarning`.

  `histogram_matching` and `interp` now return exact values:

  - `interp` returns `fp` exactly at a knot, and `np.interp`'s `5.0` instead of `6.0` left of a repeated first knot (`interp([-1, 0, 0.5], [0, 0, 1], [5, 6, 7])`).
  - Matching an image to itself is the identity (132 of 5328 `float32` and 178 of 5328 `float64` pixels used to change), `[3, 1, 2]` matched onto `[10, 20, 30]` gives `[30, 10, 20]` instead of `[29.999998, 10, 20]` (which became `29` as `uint8`), and a `float32` result no longer leaves the template's range by one ulp.
  - An empty source returns an empty tensor of the same shape instead of raising `IndexError`.
  - `interp` keeps non-finite `fp` values as `np.interp` does: `interp([0.5, 1, 1.5, 2], [0, 1, 2], [3, 4, inf])` gives `[3.5, 4, inf, inf]`, where it used to give `[3.5, nan, inf, inf]`.
  - `interp` returns `nan` for a `nan` query on a repeated last knot, as `np.interp` does: `interp([nan], [0, 1, 1], [5, 6, 7])` gave `7.0`.
  - `interp` subtracts integer coordinates in `int64`: `uint8` `x = [0, 3]` with `xp = [1, 2]` and `fp = [10, 20]` gives `[0, 30]` instead of the wrapped `[5120, 2590]`.

  ([#5960](https://github.com/kornia/kornia/pull/5960))
- `connected_components` accepts batches that cannot be viewed as one block (for example permuted leading dimensions) and inputs with `H = 0` or `W = 0`, which used to crash.

  `KMeans.fit` with `tolerance=0.0` now stops at the fixed point instead of never returning, and a second `fit` samples fresh initial centres from its own data (it reused the first fit's centres, and raised for data of another dimension) unless explicit `cluster_centers` were given. Distances in `fit` and `predict` are computed in at least `float32` (and in the promoted dtype of data and centres), so `float16` squared distances no longer overflow to `inf` and pick cluster 0. Integer data against integer centres is handled in `int64`: it was compared and summed in the narrow integer dtype, which wraps (`uint8` data: the mean of 200 and 210 came out as `[77, 77]`), and the convergence shift wrapped the same way, so the fit could stop while the centres were still moving. `predict` checks that its input is 2-D.

  `Lambda` names the right argument (`func`) in its non-callable error message; it used to name a non-existent `lambd` argument. ([#5961](https://github.com/kornia/kornia/pull/5961))
- `kornia.contrib.diamond_square`, and so `RandomPlasmaBrightness`, `RandomPlasmaContrast` and `RandomPlasmaShadow`, now generate an unbiased map. The seed divided its corners and centre by three, took the wrong corner for the right-edge midpoint and never interpolated the interior row-1 points of an elongated seed, and the square step at the border over-weighted a neighbour. A 9x9 map averaged 0.167 at the corners and the centre, 0.611 at columns 3 and 5 of the border rows and columns, and 0.39 to 0.48 in the interior (row 0: `[0.167, 0.503, 0.500, 0.611, 0.500, 0.611, 0.500, 0.503, 0.167]`), so a plasma brightness or shadow map darkened the corners and the centre of every image; the expected mean map is now a flat 0.5 for `torch.rand` with `random_scale=1`, and an unnormalized map stays in [0, 1] for `torch.rand` when `random_scale` and `roughness` are in [0, 1]. The diamond and square centres are chosen by index parity, so a signed `random_fn` such as `torch.randn` no longer leaves most pixels at zero; a tensor `roughness` or `random_scale` follows the requested `dtype` and `device`; and an integer `normalize_range` such as `(0, 1)` is accepted. The random numbers are drawn in a different order, so a fixed seed gives a different map than before. ([#5964](https://github.com/kornia/kornia/pull/5964))
- `HomographyTracker.set_target` now extracts the target's features once for `LoFTR` and `LocalFeatureMatcher` matchers and reuses them for every frame, where they were extracted again on every frame; a matcher replaced afterwards, or a target assigned directly, gets the target extracted again, once, where a custom `extract_features` matcher was passed the replaced matcher's or the old target's cached entries. The target is extracted without gradients, so a `LocalFeatureMatcher` tracker no longer passes gradients to the target, as it did when it extracted the target on every frame. `set_target`, or a target assigned directly, also starts a new track, where the previous homography, which maps the old target's coordinates, was kept. Fewer than 4 matches count as a failed match instead of reaching RANSAC, which raised. `ImageStitcher` warps its validity mask bilinearly, so a sub-pixel shift no longer leaves a dark seam or an extra dark column, and half images are warped in float32 internally (transiently), so half panoramas on MPS lose their dark border pixels and float32 keypoints with half images no longer raise. `kornia.contrib.audit` accepts `(B, H, W)` masks, which raised a `ValueError` saying the inputs must be batched. ([#5965](https://github.com/kornia/kornia/pull/5965))
- Float16 and bfloat16 `LoFTR`, `DISK`, `ALIKED`, `ScaleSpaceDetector`, `MultiResolutionDetector` and `SIFTFeatureScaleSpace` (both backends), with `KeyNetDetector` and the `LocalFeature` presets built on them, now return float32 keypoints and LAFs, formed in float32 where they are computed; scores, responses, confidences and descriptors keep the model's dtype. In bfloat16 the coordinates were rounded to a 2 px grid above x = 256 and a 4 px grid above 512, and in float16 to a 0.25 px grid above 256. A half-precision module fed these coordinates directly has to cast them: `LightGlue` now accepts float32 keypoints with a float16 or bfloat16 model (it raised `expected m1 and m2 to have the same dtype`), and `LocalFeatureMatcher` returns the same keypoint and LAF dtype whether or not a pair matches (it returned half keypoints and LAFs when no pair matched and float32 ones otherwise). `ImageStitcher` with a float16 matcher and `estimator="vanilla"` could give a NaN panorama, because the DLT ran on float16 keypoints; it now runs on float32 keypoints. ([#5966](https://github.com/kornia/kornia/pull/5966))
- `RandomJPEG` accepts an empty batch and returns it unchanged. It raised
  `ValueError: Invalid input tensor, it is empty.` before the codec ran, because the differentiable JPEG codec rejects
  an empty input. ([#5972](https://github.com/kornia/kornia/pull/5972))
- When the samples `x1` and the bin centres of `histogram2d` promote to float16 or bfloat16, it now evaluates them in float32 and casts the result back to that dtype. In float16 the joint histogram of an ordinary input, such as a 128x128 image or 1024 samples with a 0.2 bandwidth, used to come back all zero or NaN; the histogram and its gradient now follow a float64 reference. ([#5973](https://github.com/kornia/kornia/pull/5973))
- `spatial_gradient` and `sobel` now differentiate an integer or bool image in float32 instead of raising, matching
  `spatial_gradient3d`. They cast the Sobel kernel to the input dtype before, so in `uint8`, `int16` or `int32` the
  call failed, and with `normalized=False` an integer accumulation wrapped silently instead. ([#5974](https://github.com/kornia/kornia/pull/5974))
- `kornia.io.write_image` rejects an image whose height or width is zero with a `ShapeError` naming the shape,
  before the file is created. It used to reach kornia-rs, which raised a bare `Exception` and left a truncated file
  behind (#5770). ([#5975](https://github.com/kornia/kornia/pull/5975))
- `find_homography_lines_dlt` and `find_homography_lines_dlt_iterated` return an empty `(0, 3, 3)` batch for an empty batch of line segments, as `find_homography_dlt` does, instead of raising a `reshape` error. ([#5976](https://github.com/kornia/kornia/pull/5976))
- `StereoCamera.reproject_disparity_to_3D` and `kornia.geometry.camera.stereo.reproject_disparity_to_3D` return an empty `(0, H, W, 3)` point cloud for an empty batch of disparity maps, instead of raising a `reshape` error. An empty rig has been accepted since #4281. ([#5979](https://github.com/kornia/kornia/pull/5979))
- 2D intensity augmentations whose arithmetic assumes unit-range images now clamp their input into `[0, 1]` before operating, as well as their output. This fixes the NaN that `RandomGamma` returned for a negative input and a non-integer gamma, `RandomPosterize`'s platform-dependent `uint8` conversion of out-of-range values, and the negative values `RandomPlanckianJitter` kept. `RandomEqualize` and `RandomClahe` clamp an out-of-range input instead of raising; `kornia.enhance.equalize` and `equalize_clahe` still raise. `RandomHue`, `RandomSaturation`, `RandomSnow`, `ColorJiggle` and `ColorJitter` (also with neutral or hue-only steps), and `RandomPosterize` at `bits=8` used to pass out-of-range values through and now clamp them. Range-independent augmentations (blurs, noise, channel operations, erasing, grayscale, inversion, `Normalize`, `Denormalize`, `RandomAutoContrast`) and samples skipped by the probability gate keep their values; `RandomBrightness` and `RandomContrast` keep the unclipped arithmetic with `clip_output=False`. ([#5980](https://github.com/kornia/kornia/pull/5980))
- `adjust_brightness`, `adjust_brightness_accumulative`, `adjust_contrast`, `adjust_contrast_with_mean_subtraction`, `adjust_saturation_with_gray_subtraction`, `adjust_gamma`, `adjust_sigmoid`, `adjust_log`, `posterize`, `solarize`, `sharpness`, `shift_rgb` and `jpeg_codec_differentiable` raise `TypeCheckError` (a `TypeError`) for an integer image, with a message that names the conversion (`image.float() / 255`). Previously eleven of them returned a saturated, wrapped or negative result for a `uint8` image without an error, and the other two raised an unrelated PyTorch error. Floating-point input is unchanged, and the check is skipped under `disable_checks()`. ([#5983](https://github.com/kornia/kornia/pull/5983))
- The image and mask inverse of the 2D crops also changed in resample mode. With `align_corners=True`, `CenterCrop` and `RandomCrop` spread a crop with a one-pixel axis over the whole canvas axis, and now paste it back. `RandomResizedCrop` with a one-pixel output or crop axis and corner alignment returned NaN in resample mode, and now samples the crop's first row or column (or repeats a one-pixel crop), as slice mode does. `RandomCrop` inverted an oversized crop (larger than the padded canvas) without its scale, returning the top-left corner of the upscaled crop, and now scales it back. Its mask inverse raised `ValueError` when the image `fill` had one value per channel, and now removes the added padding without a fill and pads a border that negative `padding` cropped with zeros. ([#5984](https://github.com/kornia/kornia/pull/5984))
- `image_histogram2d` now returns float32 counts and pdf for float16 and bfloat16 images, with automatic centers or centers of the image dtype, instead of the image dtype; cast the result if you need the image dtype. In float16 a bin count above 65504, such as the black half of a 512x512 image, used to become `inf` and its pdf `NaN`, and bfloat16 counts were rounded (66048 instead of 65998). The 1-D `histogram` now evaluates float16 and bfloat16 kernels in float32 and still returns the input dtype: at 65536 samples, 64 bins and bandwidth 0.01 its float16 gradient had a 34 % relative error against float64 and now follows it. ([#5988](https://github.com/kornia/kornia/pull/5988))
- `median_blur` and `MedianBlur` filter an integer or bool image in float32 and return float32 on its own scale, like `box_blur`, instead of raising a dtype error. ([#5989](https://github.com/kornia/kornia/pull/5989))
- `find_homography_lines_dlt` and `find_homography_lines_dlt_iterated` raise `AssertionError` for fewer than four line segments, as `find_homography_dlt` does for fewer than four points. Previously they returned an arbitrary homography from the under-determined system (all NaN for zero segments) without an error. ([#5993](https://github.com/kornia/kornia/pull/5993))
- LightGlue now uses math attention for CPU float64 inputs on Windows with PyTorch 2.5, avoiding corrupted attention outputs, premature stopping, and incorrect point pruning. ([#5994](https://github.com/kornia/kornia/pull/5994))
- `find_homography_lines_dlt_iterated` accepts unbatched `(N, 2, 2)` line segments and treats them as `B = 1`, as `find_homography_lines_dlt` does. Previously it solved once and then raised a `ShapeError` from `line_segment_transfer_error_one_way` in its re-weighting step. ([#5996](https://github.com/kornia/kornia/pull/5996))
- Reduce scalar gamma adjustment overhead, batch confusion-matrix accumulation, skip settled distance-transform inference iterations, avoid dense one-hot intermediates for unweighted reduced focal loss, and reuse sampling grids for shared non-trainable rotation and affine transforms. Preserve TorchScript gamma adjustment and dynamic image/angle batches in rotation traces. Keep real-mode FX capture static and preserve affine-warp empty-batch validation in traces. ([#5997](https://github.com/kornia/kornia/pull/5997))
- `Rescale` works with `interpolation="nearest"`, `"nearest-exact"` and `"area"`. Previously its default `align_corners=True` was passed to `torch.nn.functional.interpolate`, which rejects any value but `None` for these modes, so every call that did not pass `align_corners=None` raised `ValueError`. `align_corners` now applies to the `bilinear` and `bicubic` modes only, whose results are unchanged; the other modes ignore it, an explicit `align_corners=True` included. ([#6010](https://github.com/kornia/kornia/pull/6010))
- `AugmentationSequential` now propagates all three output spatial dimensions after `CenterCrop3D` and `RandomCrop3D`, so later transforms generate parameters for the volume they actually receive. Previously, later 3D crops, `RandomAffine3D` and `RandomPerspective3D` drew their parameters for the original D/H/W even for nonempty batches: a later crop could read outside the cropped volume and return zero-filled voxels, and one larger than the cropped volume returned such a zero-filled result where it now raises the same `ValueError` as a direct call. Statically applied empty crops also propagate their configured size. This corrects newly generated parameter sequences; previously saved coordinates and shape history are not recalculated. Part of #4429. ([#6011](https://github.com/kornia/kornia/pull/6011))
- `resize`, `Resize`, `rescale` and `Rescale` raise `ValueError` for `interpolation="linear"` and `"trilinear"`, with a message that names the supported modes, and so do `resize_to_be_divisible`, and `crop_by_indices` when a crop has to be resized. Their docstrings no longer list these two modes. Every input is resampled as a 4-D `(B, C, H, W)` batch, which `torch.nn.functional.interpolate` rejects for them, so these modes previously raised its raw `NotImplementedError` ("Got 4D input") for every input; only a target size with a zero side returned an empty tensor, and it now raises too. The other modes are unchanged. ([#6013](https://github.com/kornia/kornia/pull/6013))
- The weighted builders of `kornia.models` return the model in evaluation mode: `Sam.from_config` with `pretrained=True` or a `checkpoint` (and so `VisualPrompter` with such a config), `RTDETR.from_pretrained`, `RTDETR.from_config` with a `checkpoint`, `TinyViT.from_config` with `pretrained` and the checkpoint's class count, `EfficientViT.from_config`, `VisionTransformer.from_config(pretrained=True)`, `SigLip2Builder.from_pretrained_hf` and `KimiVLBuilder.from_pretrained_hf`. They returned training mode, so MobileSAM, RT-DETR, TinyViT and EfficientViT normalised with batch statistics, an image's output depended on the rest of the batch, and inference overwrote the BatchNorm running statistics; TinyViT-11M and -21M also dropped residual paths at random. Randomly initialised models stay in training mode, and so does a pretrained TinyViT whose classification head is reset to zeros because `num_classes` differs from the checkpoint's, including `pretrained="in22k"` (a 21841-class head) with the default `num_classes=1000`: pass `num_classes=21841`, or call `.eval()` before extracting features. Code that fine-tunes a model returned by one of these builders must now call `.train()` first, or BatchNorm keeps its running statistics frozen and, in TinyViT-11M and -21M, DropPath stays off. After loading your own weights, call `.eval()`. TinyViT's attention bias is no longer cached at `eval()`, so `load_state_dict`, a parameter update, or `.to()` / `.half()` / `.bfloat16()` after `eval()` no longer gives a stale bias (wrong logits) or a device/dtype error. The cache was filled without gradient, so in evaluation mode `attention_biases` received none, and fine-tuning with BatchNorm frozen through `.eval()` silently froze them; they now get gradients. ([#6014](https://github.com/kornia/kornia/pull/6014))
- `LightGlue` built with complete learned weights (a `features` name, or a local `weights` file) now comes back in evaluation mode, like the other `kornia.feature` models built with pretrained weights. Missing derived buffers do not prevent evaluation mode, but checkpoints missing learned parameters leave the model in training mode, as does `LightGlue(features=None)` without weights. Its output does not depend on the mode. ([#6017](https://github.com/kornia/kornia/pull/6017))
- `equalize_clahe` and `RandomClahe` now count, clip and accumulate the tile histograms of float16 and bfloat16 images in float32 on the default (fast) path, and cast only the finished lookup tables back. Before, with clipping enabled, a float16 image whose tiles hold about 65520 or more pixels, such as 2048 x 2048 with the default 8 x 8 grid, came back entirely NaN. Once the per-bin cap `clip_limit * tile_pixels // 256` passed 65504, `equalize_clahe` raised an overflow error on a float16 image instead (and aborted the process on MPS); at the default `clip_limit=40.0` that is any tile above about 419,000 pixels, such as the 8 x 8 grid of a 7680 x 4320 image. Smaller half-precision tiles got lookup tables one or more gray levels off in float16 and several in bfloat16. The lookup tables now equal those of a float32 image that holds the 8-bit levels the lookup reads, `(x * 255).long()` in the image dtype. float32 and float64 results, and the `slow_and_differentiable` path, are unchanged. ([#6018](https://github.com/kornia/kornia/pull/6018))
- `normalize_min_max` promotes integer and boolean images to torch's default floating dtype before subtracting, as `normalize` and `denormalize` already do. A signed image such as an `int8` one holding `-128` and `127` used to wrap and return values outside the requested range, and a boolean mask raised. ([#6022](https://github.com/kornia/kornia/pull/6022))
- `warp_affine`, `warp_perspective` and `RandomAffine` with `padding_mode="fill"` keep the input's channel count. A `fill_value` that does not broadcast to the warped image without changing its shape now raises a `ValueError` that names both shapes. Previously it was broadcast against the image regardless, so a one-channel image with a three-value fill came back with three channels, and other mismatches raised a raw broadcasting error. Fills that do broadcast, such as a per-sample `(B, C, 1, 1)` or a per-pixel `(B, C, H, W)` fill, give the same results as before. `warp_perspective` now defaults to one zero per channel and accepts the same fills as `warp_affine`. Previously it substituted `zeros(3)` and required a three-value fill, so a one-channel image gained channels even without a `fill_value`. In `AugmentationSequential`, a per-channel `RandomAffine` `fill_value` pads masks with zero unless the mask sets its own `fill_value`, as a per-channel `fill` already does, both when augmenting and when inverting. Matching fills give the same results as before. ([#6023](https://github.com/kornia/kornia/pull/6023))
- `torch.export` of an `AugmentationSequential` holding a `Resize` with an integer `size`, a `LongestMaxSize` or a `SmallestMaxSize` no longer fails in the container's shape tracking. The integer side length is now converted to the aspect-preserving output size, as the eager path already does, so a later shape-dependent transform such as a resample-mode crop sees the shape it actually receives. ([#6027](https://github.com/kornia/kornia/pull/6027))
- `focal_loss` and `FocalLoss` accept a one-element tensor `alpha` again with `reduction="mean"` or `"sum"`. Since #5997, a `(1,)` or `(1, 1)` tensor, or a 0-d one that requires grad, raised `TypeError: new_full(): argument 'fill_value' ... must be Number, not Tensor` while `reduction="none"` accepted it, and `torch.compile(fullgraph=True)` failed for every tensor `alpha`. An `alpha` that is not a Python number takes the dense path again, as before #5997; a Python-number `alpha` keeps the reduced path. ([#6028](https://github.com/kornia/kornia/pull/6028))
- The model `DepthAnythingONNXBuilder.build` returns resizes each image to the size transformers' `DPTImageProcessor` computes from the model's `preprocessor_config.json`: it keeps the aspect ratio, fitting the side whose scale is closer to 1, and makes each side a multiple of 14. The filter is an antialiased bicubic, the one of transformers' DPT image processor (its torchvision backend, `torchvision.transforms.v2.functional.resize` with `antialias=True`), which approximates PIL's resize. The depth is resized back to the image size bicubically, as transformers' `post_process_depth_estimation` does; it was resized back bilinearly. Every image used to be squashed to 518 x 518 with a bilinear resize without antialiasing, so a 16:9 frame was squeezed 1.78x horizontally before the network saw it; a 300 x 400 image now reaches the network at 392 x 518, and a 1080 x 1920 frame at 518 x 924. Every image other than 518 x 518 is now resized by this rule and these filters, so its depth changes. The config parser behind it reads `keep_aspect_ratio`, `ensure_multiple_of` and `resample` (PIL's nearest, bilinear or bicubic), with the processor's defaults (`False`, `1`, bicubic) when they are absent, and raises `NotImplementedError` for any other `resample` filter, which it used to replace with bilinear. `DepthEstimation` accepts images of any floating dtype: it runs the graph in float32 and returns the depth in the image's dtype, where a float64, float16 or bfloat16 image raised inside ONNX Runtime or NumPy. An integer image raises kornia's `BaseError` instead of ONNX Runtime's `InvalidArgument`. `DepthEstimation.visualize` and `save` accept a tensor batch, on which they raised `ValueError`. ([#6030](https://github.com/kornia/kornia/pull/6030))
- The mix augmentations (`RandomMixUpV2`, `RandomCutMixV2`, `RandomMosaic`, `RandomJigsaw`, `PatchMix`, `RandomTransplantation` and `RandomTransplantation3D`) raise the documented `RuntimeError: Inverse for <Class> is not supported.` from `inverse(output)` and `inverse(output, params=...)`. Previously only a call without positional arguments did; these call forms raised `TypeError: MixAugmentationBaseV2.inverse() takes 1 positional argument but 2 were given`. ([#6033](https://github.com/kornia/kornia/pull/6033))
- The `kornia.morphology` operators run about 2x faster with less memory on CUDA, and with `engine="unfold"` on CPU, for a kernel without `structuring_element` when no gradient is recorded: the `unfold` engine masks the windows once and reduces them with `amax`/`amin`, without indices. Values are unchanged; as on every engine, an `erosion` window that ties `+0.0` with `-0.0` may return either zero. Calls that record gradients, forward-mode AD, `torch.func` transforms and traces keep the previous reduction. ([#6034](https://github.com/kornia/kornia/pull/6034))
- Detection, processor and super-resolution wrappers of `kornia.models` and `kornia.contrib` return correct results:

  - `DETRPostProcessor` scales each image's boxes by that image's own row of `original_sizes`. It used the first image's size for the whole batch: with sizes `[[100, 200], [400, 800]]` (H, W), the normalised box `(0.5, 0.5, 0.25, 0.25)` of image 1 came back as `[75, 37.5, 50, 25]` (xywh) instead of `[300, 150, 200, 100]`, so `ObjectDetector` on images of different sizes misplaced the boxes of every image but the first. The boxes also keep their own dtype when the sizes have another one.
  - `RTDETRDetectorBuilder.build(confidence_filtering=False)` builds a post-processor that does not filter and returns the `(B, Q, 6)` tensor; the flag was ignored and the detections were filtered into a list. `None`, the default, still filters.
  - `ResNetD` runs on inputs that give an odd feature map, such as 636 x 636, which raised `RuntimeError: The size of tensor a (80) must match the size of tensor b (79)`: the stride-2 shortcut pools with `ceil_mode=True`, as in the reference `presnet.py`. `RTDETR` then runs wherever its neck can join the three feature levels, as in the reference: 636 x 636 runs, while 600 x 600 still stops in the neck. Inputs at multiples of 32 give the same outputs as before.
  - `rtdetr_r50vd_m` decodes with the third of its six decoder layers (`eval_idx: 2` in the reference configs of that checkpoint), as the published R50-M model does; it ran all six, which moved its logits by up to 3.5 on a sample image. `RTDETRConfig` gains `eval_idx` (`None` selects the variant's layer) and `RTDETRHead` an `eval_idx` argument; an index outside the decoder raises.
  - `ResizePreProcessor` records the original sizes as `int64` on the images' device. It stored them in the image dtype, so a `bfloat16` 667 x 1000 image was recorded as 668 x 1000 and a `float16` 4097 x 2051 image as 4096 x 2052, and `ResizePostProcessor`, as in `EdgeDetector`, returned a map one row too tall.
  - `output_type="pil"` in the `visualize` of `ObjectDetector`, `EdgeDetector`, `SuperResolution`, `SemanticSegmentation` and `DepthEstimation` returns a list of `PIL.Image.Image`, one per `(C, H, W)` or `(H, W)` image; it returned NumPy arrays.
  - `ModelBase.load_checkpoint(path, device=...)` moves the model to `device`; the parameters stayed where they were, on the CPU after `device=torch.device("mps")`.
  - `SuperResolution` built by `RRDBNetBuilder` runs a list of images one at a time; a list, even of one image, raised `RuntimeError: Sizes of tensors must match except in dimension 1`.
  - `save(output_type="pil")` raised in `SemanticSegmentation`, `EdgeDetector`, `SuperResolution` and `DepthEstimation` (`SemanticSegmentation`: a `ValueError` for a batch tensor and a `TypeError` for a list of images; the others: an `AttributeError` on the NumPy arrays). It now writes the same files as `output_type="torch"`, with the same pixels.

  ([#6035](https://github.com/kornia/kornia/pull/6035))
- `KeyNet` and `KeyNetDetector` build their learnable block with `num_filters` channels per level, so a `num_filters` other than the default 8 no longer crashes in `forward`. The default architecture and the pretrained weights are unchanged. ([#6036](https://github.com/kornia/kornia/pull/6036))
- `HyNet(pretrained=True, is_bias_FRN=False)` now raises `ValueError` before downloading the weights. Previously it loaded the checkpoint without error but left the trained FRN biases unused, so its descriptors differed from the pretrained ones. Pass `pretrained=False` to build `HyNet` with `is_bias_FRN=False`. ([#6037](https://github.com/kornia/kornia/pull/6037))
- `guided_blur` and `GuidedBlur` now compute window statistics for float16 and bfloat16 images in float32. With scalar regularization, coefficients are also computed in float32 and only the result is cast back, so a half-precision call returns the float32 result for the same pixel values, rounded once. A float64 tensor `eps` widens coefficient arithmetic and the grayscale result to float64; multichannel guidance solves in float64 before casting its coefficients back. Before, the local variance `E[I^2] - E[I]^2` was formed in half precision, whose rounding at mid-gray (about 2e-4 in float16, 2e-3 in bfloat16) swamped a small `eps`: with `eps=1e-4` on two 512 x 512 test photographs, float16 results were off by up to 1.5 with grayscale guidance and 36 with RGB guidance, gradients came back non-finite, and with RGB guidance `torch.linalg.solve` could fail on a singular matrix. A float16 or bfloat16 multichannel guidance with an `input` of another dtype (float32, uint8 or the other half-precision type) raised a dtype mismatch; it now returns float32. Autocast is now disabled inside the filter: a float32 call in an autocast region used to be filtered in the autocast dtype, with the same errors, and returned that dtype with RGB guidance; it now returns the float32 result. float32 and float64 results outside autocast are unchanged. ([#6038](https://github.com/kornia/kornia/pull/6038))
- `VisualPrompter.predict(boxes=...)` uses a box prompt given as a tensor; it was dropped, and the prediction was the one without prompts. A `(1, K, 4)` box prompt is read as the K boxes of the image; for K > 1 it raised a tensor size mismatch, and box prompts for more than one image now raise a clear error instead of a batch size mismatch. `VisualPrompter` resizes the image as SAM's reference does, with `torch.nn.functional.interpolate(image, size, mode="bilinear", align_corners=False, antialias=True)`, and keypoint prompts follow the same half-pixel map: it resized with `align_corners=True` while the masks are resized back with `align_corners=False`, which shifted the masks by up to 0.48 original pixels, and it did not antialias when it downscaled. On a 1380 x 2076 image, the mask IoU against SAM's own pipeline goes from 0.957, 0.966 and 0.943 to 0.969, 0.969 and 0.968. Box prompts are mapped as SAM's `apply_boxes` maps them: kornia's `xyxy` corners are pixel edges and scale as `x * s`, so the box `(0, 0, W, H)` covers the whole resized image; they went through the `Boxes` vertex at `x2 - 1` and the corner-aligned map, which put the box `(10, 20, 60, 50)` of a 77 x 128 image at `(80.55, 161.84, 476.25, 397.51)` instead of `(80, 160, 480, 400)`. `VisualPrompter.predict` and `Sam.forward` run in float16, bfloat16 and float64, with or without prompts: the SAM prompt encoder built its positional encoding in float32, and they raised `RuntimeError: expected m1 and m2 to have the same dtype`; it now computes the encoding in float32 (float64 for a float64 model), with autocast disabled, and casts it to the dtype of the model. Keypoint and box prompts are mapped to the resized image in float32 (float64 for a float64 model) and handed to the prompt encoder in that dtype, which also shifts them to pixel centres in that dtype. `kornia.models.common.LayerNorm2d` overflowed in float16 once a value was more than 255.9 from its channel mean, so pretrained MobileSAM gave empty masks in float16; float16 and bfloat16 inputs are now normalised in float32. `Sam.from_config` builds a custom ViT image encoder from `SamConfig(encoder_embed_dim=..., encoder_depth=..., encoder_num_heads=..., encoder_global_attn_indexes=...)` with `encoder_global_attn_indexes` a tuple, a list or an int; it raised `NotImplementedError` for a tuple or a list and `TypeError` for an int. A custom encoder config with `pretrained=True` and no `model_type` raises a `BaseError` saying that the released weights need a `model_type`; it raised `KeyError: None`. ([#6040](https://github.com/kornia/kornia/pull/6040))
- `Image.from_dlpack` takes `color_space` and `channels_order` arguments, as `Image.from_numpy` does, so a channels-last `(H, W, C)` capsule is read with the right height, width and channels, and `Image.from_dlpack(img.to_dlpack(), img.pixel_format.color_space, img.channels_order)` round-trips any image. Previously `from_dlpack` always read the capsule as `(C, H, W)` and labelled it by its channel count, so the round trip of a channels-last image, the `from_numpy` default, silently swapped its height, width and channels, and a `BGR` image came back labelled `RGB`. The defaults are unchanged, and `to_dlpack` now documents that the capsule keeps the image's own layout. ([#6041](https://github.com/kornia/kornia/pull/6041))
- `FaceDetector` flips its input from RGB to BGR before YuNet, whose weights were trained on BGR images. It fed RGB, which found far fewer faces: run on the docs figure `yunet.jpg` as rendered before this change (421×1185, input and output panels together), the detector now returns 600 detections instead of 221, and 225 instead of 50 above a score of 0.6. The re-rendered figure's title reports 320 faces where it reported 228. The input stays RGB in [0, 255]; the bare `YuNet` takes BGR in [0, 255]. A caller that already passed BGR, for example a `cv2.imread` array, must now pass RGB, since the flip happens inside. An input that is not `(*, 3, H, W)`, such as a 1- or 4-channel or a channels-last image, raises a `ValueError` naming that shape from the flip; it raised a `RuntimeError` from inside YuNet. `EdgeDetectorBuilder` feeds DexiNed BGR in [0, 255] minus the BGR mean `[103.939, 116.779, 123.68]`, the input it was trained on, instead of RGB minus the ImageNet RGB mean; on two street photos this changed 8–11 % of the `p > 0.5` edge decisions. The input stays RGB in [0, 1]. `DexiNed` (`kornia.models.dexined` and `kornia.filters.dexined`) no longer raises a `RuntimeError` about an in-place operation in backward; its forward output is unchanged. `FaceDetectorResult` checks the last dimension of its data, so `(15,)` and `(N, 15)` are accepted and a row of fewer than 15 values raises `ValueError`. It checked the first dimension, so a `(3, 15)` tensor raised and a `(20, 14)` one was accepted until `.score` raised `IndexError`. ([#6042](https://github.com/kornia/kornia/pull/6042))
- The vision-language models of `kornia.models` follow their reference implementations in more places:

  - `SigLip2Model(..., return_loss=True)` returns SigLIP's pairwise sigmoid loss over every image-text pair of the batch, with the matching pairs labelled +1 and every other pair -1. It used only the matching pairs, so the negatives never entered and the loss fell to zero as `logit_bias` grew: `0.0` on a batch of 4 with `logit_bias = 10`, where SigLIP's loss is `38.54`. The loss pairs the i-th image with the i-th text, so unequal image and text batches raise `BaseError` with `return_loss=True`: 3 images with 1 text raised `IndexError`, and 1 image with 3 texts returned the loss of its first pair only. Without the loss, unequal batches still give rectangular logits.
  - The `attention_mask` of `SigLip2VisionModel` and `SigLip2Model.get_image_features` reaches the attention-pooling head as well as the encoder, so the pixels of masked patches no longer change the pooled embedding. For a `(B, N, N)` or `(B, 1, N, N)` mask, the head leaves out the patches no query may attend to.
  - An image whose patches are all masked is pooled by `SigLip2VisionModel` with uniform weights over its patches, as transformers' pooling mask gives: it adds `torch.finfo(dtype).min` to the masked scores, not `-inf`. The weights are exactly uniform in float32, float64 and bfloat16. In float16, as in transformers, `finfo(float16).min` plus a score rounds to a multiple of 32, so they are only roughly uniform, and a head whose scores are all below about -16 gives a non-finite output.
  - A `(B, N, C)` patch output of `NaFlex`'s patch embedding carries no grid and is read as a square grid, which is silently wrong for a non-square grid (128 x 512 at patch 16) and an error for a token count that is not a square number (192 x 336): pass the new `patch_size` argument, which gives such an output the grid `(H // p_h, W // p_w)`, or return patch embeddings of shape `(B, C, h, w)`. Without `patch_size`, the square reading is unchanged, and a token count that is not a square number raises `BaseError` saying so, where it raised a shape error or gave wrong values. A float16 or bfloat16 table on the CPU is resized in float32. A position table passed as an `nn.Parameter` stays a parameter of the module and is trained with it; it was registered as a buffer.
  - `KimiVLModel` and `MoonViT` apply a `(B, N, N)` attention mask per sample, shared by the heads. The mask's leading dimension lined up with the attention heads: outputs were wrong when the batch size equalled the number of heads, and batch sizes other than 1 and the number of heads raised.
  - When the checkpoint of a repository outside its supported list does not fit the configuration `SigLip2Config.from_name` reads from the name, `SigLip2Builder.from_pretrained_hf` raises a `RuntimeError` that gives that configuration (hidden size, layers, heads, patch size, image size, vocabulary) and lists the supported names. It raised `load_state_dict`'s bare size-mismatch error. A fine-tune or mirror whose checkpoint fits the configuration read from its name loads as before. An image size in such a name counts only as a whole number, of the repository part after the last `/` and else of the owner part: `org2560/siglip2-base-patch16-384` and `org-256/siglip2-base-patch16-384` read 384 where they read 256, and `x/siglip2-base-patch16-ckpt2560` reads 224 where it read 256, while `siglip384/finetune` still reads 384.

  ([#6044](https://github.com/kornia/kornia/pull/6044))
- Two corrections in `kornia.contrib`:

  - `kornia.contrib.distance_transform` and `DistanceTransform` clamp the mask to [0, 1], so a mask with values above 1, such as 0/255, no longer collapses the result to all zeros: every value above 1 counts as a full source, and negative values as background. Mask values outside [0, 1] now get a zero gradient; they had one before, when the mask was used as given. Values in [0, 1], the bounds included, keep theirs.
  - `kornia.contrib.diamond_square` with `random_scale` other than 1 no longer scales the seed by `random_scale`. The expected value is 0.5 at every pixel with `torch.rand`; it was 0.5·`random_scale` at the seed nodes and moved towards 0.5 level by level elsewhere (0.25 vs 0.31–0.35 at 0.5). `random_scale=1`, the default used by `RandomPlasmaBrightness`, `RandomPlasmaContrast` and `RandomPlasmaShadow`, gives the same output as before. With `dtype=None`, a float64 tensor `roughness` now gives a float64 map at every size, as a float64 tensor `random_scale` already did; with only `roughness` in float64, a map with a side of 3 px or less came out float32 and a larger one float64.

  ([#6045](https://github.com/kornia/kornia/pull/6045))
- `ImageSequential` and `AugmentationSequential` accept integer, boolean and tensor `random_apply_weights`. Previously integer or boolean weights raised `RuntimeError: multinomial only supports floating-point dtypes` on every call, and a tensor of weights raised `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` in the constructor. Non-floating weights are converted to the default floating dtype, and float weights keep their dtype, so existing selections are unchanged. Weights that are not one-dimensional now raise a `ValueError`; a nested float list used to be accepted and drew a selection per row, so `random_apply=1` applied more than one transformation. ([#6050](https://github.com/kornia/kornia/pull/6050))
- More vision-language fixes in `kornia.models`, following #6044:

  - Under `torch.autocast`, an image whose patches are all masked no longer pools to `nan` in `SigLip2VisionModel`'s attention-pooling head: the head filled the masked scores with the float32 minimum, which rounds to `-inf` in float16 and bfloat16, and now builds the mask in the dtype the scores are computed in, with that dtype's minimum (a float64 model, which autocast leaves alone, keeps float64's). Such an image gets uniform weights under bfloat16 autocast, as without autocast. Under float16 autocast it gets float16's rounding, as in float16 without autocast: roughly uniform weights, and a non-finite output for a head whose scores are all below about -16.
  - `MoonViTAttention`, `MoonViT` and `KimiVLModel` read a `(B, N)` attention mask as a padding mask over each sample's keys, as `SigLip2Attention` does. It went to `scaled_dot_product_attention` as an `N x N` query x key mask: a batch size other than 1 and `N` raised a size mismatch, and `B == N` masked query rows by batch index. An integer mask, such as a tokenizer's `int64` padding mask, is read as a boolean one, where `scaled_dot_product_attention` raised for it.
  - `NaFlex` warns, naming `patch_size`, when it reads a `(B, N, C)` patch output without `patch_size` as a square grid that takes rectangular patches: 256 tokens of a `128 x 512` image, the `8 x 32` grid of #6004 at patch 16, are read as `16 x 16`. It added the table in that layout silently. A rectangular patch embedding gets the same warning, and `patch_size` silences it; a square-patch convolution padded by at most half its kernel never gets it. The warning names the line that calls the module.
  - `SigLip2Config.from_name`, `SigLip2Builder.from_name` and `SigLip2Builder.from_pretrained_hf` read names in any case: `Google/SigLIP2-Large-Patch16-384` is the supported checkpoint, which `from_pretrained_hf` downloads and caches under its listed spelling, and `someone/SigLIP2-Large-patch16-256` the large architecture, where it was read as base. A fine-tune or mirror of `google/siglip-base-patch16-224` under another owner, such as `someone/siglip-base-patch16-224-finetuned`, gets SigLIP 1's 32000-token vocabulary, where it got SigLIP 2's.

  ([#6051](https://github.com/kornia/kornia/pull/6051))
- `ScaleSpaceDetector` (`forward` and `detect`) and the patch descriptors `SIFTDescriptor`, `HardNet`, `HardNet8`, `HyNet`, `TFeat` and `SOSNet` accept an empty batch and return empty outputs of the documented shapes: responses `(0, N)` and LAFs `(0, N, 2, 3)` from the detector, descriptors `(0, D)` from the descriptors. So do `LocalFeature`, `SIFTFeatureScaleSpace` and `get_laf_descriptors` on zero images. They raised an unrelated reshape `RuntimeError`; an invalid input still raises what the same invalid non-empty input raises. A `ScaleSpaceDetector` call on one image with no detected candidates returns responses linked to autograd, so a backward through them gives the input a zero gradient; it used to raise. ([#6052](https://github.com/kornia/kornia/pull/6052))
- `scale` and `Scale` now accept one isotropic scale factor per image as a `(B,)` tensor when the batch contains more than one image, instead of raising a broadcasting error. ([#6054](https://github.com/kornia/kornia/pull/6054))
- Fixes in the vision backbones of `kornia.models`:

  - `MobileViT` raises `ValueError` for a `mode` other than `"xxs"`, `"xs"` or `"s"`, where it raised `UnboundLocalError`.
  - `TinyViT.from_config("21m", ...)` builds the `"in1k_384"` and `"in1k_512"` checkpoints, named explicitly or picked by `pretrained=True` for an `img_size` of at least 384 or 512, with the window sizes they were trained with, `[12, 12, 24, 12]` and `[16, 16, 32, 16]`, as the reference TinyViT does. It built them with the 224-px windows `[7, 7, 14, 7]`, so their 12- and 16-px attention biases were bicubic-resized to 7 px without a warning, e.g. `(6, 144)` to `(6, 49)` in stage 1. Random init and the other checkpoints keep `[7, 7, 14, 7]`.
  - `TinyViT.from_config` passes `embed_dims`, `depths`, `num_heads`, `window_sizes` and `drop_path_rate` to the model in place of the variant's values. Each of them raised `TypeError: ... got multiple values for keyword argument`.

  ([#6056](https://github.com/kornia/kornia/pull/6056))
- `ImageSequential` and `AugmentationSequential` with a `random_apply` range that includes 0, such as `(0, 2)`, `(0,)` or `0`, apply no transformation when 0 is drawn, a mix augmentation included, and return their inputs unchanged. Previously such a call raised `RuntimeError: cannot sample n_sample <= 0 samples`, so a range that includes 0 crashed at random. A negative bound, or an upper bound below the lower one, now raises a `ValueError` naming `random_apply` when the container is constructed, instead of a raw `torch.multinomial` or `torch.randint` error at call time. Ranges that never draw 0 select exactly as before. ([#6057](https://github.com/kornia/kornia/pull/6057))
- `kornia.contrib.interp` computes a `float16` or `bfloat16` result in `float32` and rounds once, so every half-precision result can differ from before (it was up to about five ulps of the larger end value off in an ordinary segment). Integer coordinates, and half-precision coordinates, with a difference above 65504 no longer overflow the weight: `interp(x=100000, xp=[0, 200000], fp=float16 [0, 2])` gave NaN, `x = 60000` and `x = 1000` gave a silent 0 where the values are 0.6 and 0.01, and `float16` coordinates `[-60000, 60000]` with `fp = [0, 2]` gave NaN or 0 where the values are in `[0, 2]`. An integer `x` or `xp` mixed with half-precision coordinates is no longer rounded onto the half-precision grid: `interp(x=3001, xp=float16 [3000, 3010], fp=float16 [-40000, 40000])` gave -40000 where the value is -32000. A segment whose value difference overflows the half dtype is within about half an ulp of its larger end value, and its knots are exact. ([#6059](https://github.com/kornia/kornia/pull/6059))
- `kornia.enhance.normalize` and `kornia.enhance.denormalize` accept a Python `int` `mean` or `std` and treat it as the equal `float`, as `Normalize` and `Denormalize` already did. `denormalize(x, 0, 255)` and `normalize(x, 0.5, 2)` raised `AttributeError: 'int' object has no attribute 'shape'`. ([#6061](https://github.com/kornia/kornia/pull/6061))
- `guided_blur` promotes a multichannel guidance to the dtype it shares with the input, so a float16, float32 or uint8 guidance next to a float64 input no longer raises `RuntimeError: expected scalar type Float but found Double`. ([#6062](https://github.com/kornia/kornia/pull/6062))
- Input checks no longer make a dynamic-shape `torch.compile` recompile for each new input size: their messages formatted tensor sizes or `int` arguments on every call, which specialized them. Over three sizes, `get_motion_kernel2d`, `motion_blur`, `fft_conv` with `padding="valid"`, `gaussian_blur2d` with a varying `kernel_size` and DeDoDe's `PatchEmbed` now compile one graph instead of three, `solve_pnp_dlt` two instead of four, `jpeg_codec_differentiable` four instead of eight (its quality-range message also read the tensor twice per call, which broke the graph and synchronized with the device in eager mode), and `ScaleSpaceDetector` 10 instead of 15. Size checks now raise `ShapeError` with both sizes, and the messages are no longer built on every call in eager mode. ([#6063](https://github.com/kornia/kornia/pull/6063))
- Fix `rotate`, `Rotate`, `scale`, `Scale`, `rotate3d` and `Affine` treating the channels of an unbatched input as a batch. Multichannel inputs now retain their shape instead of gaining an extra dimension or raising an error. ([#6065](https://github.com/kornia/kornia/pull/6065))
- `SigLip2Attention` and `MoonViTAttention`, with `MoonViTEncoder` and `MoonViT`, check the trailing dimensions of an
  attention mask: the last one, the keys, must be the sequence length, and in a 3-D or 4-D mask the one before it, the
  queries, the sequence length or 1. A mask with a key dimension of 1, such as `(B, 1)` or `(B, N, 1)`, used to be
  broadcast over every key: on the CPU and in CUDA float64 it gave the output of no mask, and in CUDA float32, float16 and
  bfloat16 `scaled_dot_product_attention` raised `RuntimeError: (*bias): last dimension must be contiguous`. A query
  dimension other than 1 or the sequence length raised a broadcasting `RuntimeError`. Both now raise a `ShapeError`, a
  `ValueError`, that names the sequence length. ([#6070](https://github.com/kornia/kornia/pull/6070))
- With `random_apply` and a mix augmentation, `ImageSequential`, `AugmentationSequential` and `VideoSequential` insert only a mix member at the mix-insertion step. That step drew among every member whose weight was 0, so an ordinary member given weight 0 in `random_apply_weights` could still be applied, and was reported as the added mix. ([#6071](https://github.com/kornia/kornia/pull/6071))
- `find_homography_dlt`, `find_homography_dlt_iterated`, `find_homography_lines_dlt`,
  `find_homography_lines_dlt_iterated`, `find_fundamental`, `run_7point`, `run_8point` and `RANSAC` run with autocast
  disabled, so a caller's `torch.autocast` region no longer changes their result. Under bfloat16 autocast a float32 input
  used to be fitted with its matrix products in bfloat16 and returned as bfloat16: the iterated DLT homography of 500
  points with 1 px noise moved the corners of a 600 x 450 image by 3 to 5 px, a `RANSAC("homography",
  local_optimization="dlt")` estimate was about 3 px off instead of 0.5 px, and `find_fundamental` returned a bfloat16
  matrix far from the float32 one. On CUDA and MPS, the 8-point `find_fundamental` with more than eight points or with
  weights, and `RANSAC("fundamental", local_optimization="dlt")` and `RANSAC("essential", local_optimization="dlt")`,
  raised `NotImplementedError` from `torch.linalg.eigh`. The default `local_optimization="lm"` solved and scored its
  minimal samples in the autocast dtype, which could change its estimate. They, and `find_affine`, enter that
  autocast-free region only inside an autocast region, so the direct solvers keep running on the `meta` device, and
  `find_affine`, which raised `RuntimeError: unsupported scalarType` there, now runs too. The 8-point `find_fundamental`,
  `RANSAC("fundamental", local_optimization="dlt")` and `RANSAC("essential", local_optimization="dlt")` with float16 or
  bfloat16 points solve their eigenproblem in float32: with more than eight points or with weights they raised
  `NotImplementedError` from `torch.linalg.eigh` outside CPU autocast and on every other device, and now return the input
  dtype. Weights of another dtype than the points take the points' dtype in `find_homography_dlt`,
  `find_homography_dlt_iterated`, `find_homography_lines_dlt` and the 8-point `find_fundamental`: autocast used to cast
  them together, so half-precision points with float32 weights fitted only inside it, and now fit in and out of it. ([#6072](https://github.com/kornia/kornia/pull/6072))
- `equalize`, `equalize3d`, `RandomEqualize` and `RandomEqualize3D` now count each pixel of a float16 or bfloat16 image in the histogram bin of the lookup-table entry it reads. The bin was computed in the input dtype, where an 8-bit level v rounds up to bin v + 1 for v >= 240 in float16 and v >= 128 in bfloat16, so neighbouring levels merged and the whole table moved: `[10, 100, 254, 255] / 255` equalized to `[0, 150, 255, 255]` instead of the `[0, 100, 200, 255]` that float32 and `PIL.ImageOps.equalize` give, and a random 64 x 64 8-bit image was up to 16 gray levels off. Half-precision results on 8-bit images now match float32 up to output rounding; float32 and float64 results are unchanged. ([#6075](https://github.com/kornia/kornia/pull/6075))
- ONNX exports that declare a dynamic batch now run at every batch size. `to_onnx` traced them from a batch of 1, which `torch.export` specializes. On torch 2.9, the exported DexiNed edge detector and SAM image encoder kept that batch inside the graph, and ONNX Runtime rejected any other. `ResizePreProcessor` looped over the batch, so the loop unrolled at the traced size. On every torch release, the default exports of `EdgeDetector` and `SmallSR`, which include that pre-processor, returned one image's output for a batch of any size. `to_onnx` now traces each dynamic dimension at a size of at least 2, and `ResizePreProcessor` resizes a tensor batch in one call. An `ObjectDetector` whose post-processor returns a list of per-image detections, as the default confidence-filtered RT-DETR detector does, used to declare a dynamic batch and return the first image's detections for any batch. It is now exported for a batch of one image; build an RT-DETR detector with `confidence_filtering=False` to export a batch of any size. ([#6076](https://github.com/kornia/kornia/pull/6076))
- With `random_apply`, `ImageSequential`, `AugmentationSequential`, `PatchSequential` and `VideoSequential` sample with replacement when the drawn count cannot be filled from the children with a nonzero weight in `random_apply_weights`, as they already did when it exceeded the sum of the weights. A count above the number of children raised `RuntimeError: cannot sample n_sample > prob_dist.size(-1) samples without replacement` whenever the weights summed to at least the count, for example `random_apply=4` with weights `[7, 2, 1]`. A count above the number of nonzero weights but not of children filled the remaining places with children of weight 0, which the caller had excluded, or with a mix augmentation outside the mix-insertion step. ([#6078](https://github.com/kornia/kornia/pull/6078))
- `bilateral_blur` and `joint_bilateral_blur` run faster on CPU float32 and float64 inputs with at least 65,536 values
  whose gradient is not recorded: the filter is accumulated tap by tap from shifted slices of the padded input instead of
  from its unfolded windows, whose eight temporaries of `kernel_size` times the image were fresh memory faulted in on
  every call. On batches of 256 x 256 images with a 5 x 5 kernel, measured on an i7-14700K under WSL2 (where page faults
  cost more than on native Linux), the RGB filter is 3.0-3.2x faster at batch 8 and 2.8-3.0x at batch 32 in eager mode,
  and the grayscale one 17-20x and 13-17x; with the faults removed from both paths by `GLIBC_TUNABLES`, the speed-ups are
  1.8x and 1.75x (RGB) and 8.7x and 8.0x (grayscale). Under `torch.compile` the RGB filter is 1.06x and the grayscale one
  2.9x faster; kernels of more than 25 taps keep the unfold path there, because the compiled tap loop takes longer to
  compile with every tap. The values agree with the unfold path to rounding. Exported and traced programs, reverse-mode
  gradients, CUDA, MPS and half precision keep the unfold path. ([#6079](https://github.com/kornia/kornia/pull/6079))
- `ssim` and `SSIM`, and `ssim_loss` and `SSIMLoss` evaluated without a recorded gradient, run faster on the CPU for two
  contiguous images computed in float32 (float32, half-precision and integer images): the five Gaussian filterings now run
  on channels-last copies of the images, which oneDNN convolves directly, and the local moments are combined in place.
  Previously each of the ten depthwise convolutions copied its NCHW input into blocks of 8 or 16 channels and back,
  buffers 2.7 to 5.3 times an RGB batch that a large batch got as fresh memory on every call. On batches of 256 x 256 RGB
  images, measured on an i7-14700K under WSL2 (where page faults cost more than on native Linux), `ssim` is 2.9x faster at
  batch 8 and 32, and 2.0x to 2.3x at batch 8 and 1.4x to 2.0x at batch 32 with the faults removed from both paths by
  `GLIBC_TUNABLES`. The map was bitwise equal to the NCHW path's in every case measured. A call that records a gradient,
  float64, channels-last and other non-contiguous images, CUDA and MPS inputs, `vmap` and other `torch.func` transforms,
  builds without oneDNN (such as the macOS arm64 wheels) or with it switched off, and exported and traced graphs keep the
  NCHW convolutions; under `torch.compile`, a channels-last reverse pass was slower than Inductor's own layout choice. ([#6080](https://github.com/kornia/kornia/pull/6080))
- `focal_loss` and `FocalLoss` run faster on the CPU for `'mean'` and `'sum'` without class weights, on float32 and
  float64 logits with at least 2**19 values: the target-class log-probability comes from a custom autograd function that
  works through the batch in chunks and saves only the logits and two values per pixel, instead of a full-size
  `log_softmax` whose output and gradients were fresh memory faulted in on every call. On 21-class 256 x 256 logits,
  forward plus backward, measured on an i7-14700K under WSL2 (where page faults cost more than on native Linux), the loss
  is 1.9x faster at batch 8 and 1.8x at batch 32 in eager mode, 1.4x at both with the faults removed from both paths by
  `GLIBC_TUNABLES`, and 1.16x at batch 8 under `torch.compile`. Values and gradients agree with `log_softmax` to rounding,
  with the same NaN and infinity positions; double backward is supported, and exported, traced, autocast, functorch and
  forward-mode AD calls keep `log_softmax`. ([#6081](https://github.com/kornia/kornia/pull/6081))
- `dice_loss` and `DiceLoss` run faster on the CPU for float32 and float64 logits with at least 2**15 values (and class
  weights, if any, of the logits' dtype): the loss no longer builds a one-hot target and full-size products of it. The
  intersection takes each pixel's target-class probability, the micro average sums the (weighted) probabilities over the
  classes of each pixel, and the macro average gets each class's probability mass from a batched matrix-vector product and
  adds the per-class sums into (batch, class) bins. On 21-class 256 x 256 logits, forward plus backward, measured on an
  i7-14700K under WSL2 (where page faults cost more than on native Linux), the default micro loss is 4.5x faster at batch
  8 and 4.2x at batch 32 in eager mode, the macro loss 3.3x and 3.1x and the class-weighted micro loss 3.1x and 2.9x; with
  the faults removed from both paths by `GLIBC_TUNABLES` they are 2.9x to 4.3x faster, and 1.1x to 1.2x under
  `torch.compile` on torch 2.14. CUDA is unchanged. Values and gradients agree with the one-hot sums to rounding, with the
  same NaN positions, and labels outside the classes raise `one_hot`'s errors, also under `torch.compile`; exported,
  traced, autocast and functorch calls, and smaller inputs, keep the one-hot target. So do compiled calls on torch
  releases that report a `torch.compile` capture as an export (2.5.1 and 2.9.1 among them), and there a compiled call does
  not check the labels, as on main. ([#6082](https://github.com/kornia/kornia/pull/6082))
- `geometry.bbox.nms` computes half-precision and narrow-integer box areas in a wider dtype, preventing duplicate boxes from surviving or distinct boxes from being suppressed because areas or unions overflow on image-sized coordinates, and bfloat16 rounding from moving an IoU across the threshold. ([#6083](https://github.com/kornia/kornia/pull/6083))
- Augmentations no longer synchronize with the GPU to move their parameters in eager mode: parameters sampled on the CPU
  and per-call constants reach CUDA inputs without a `cudaStreamSynchronize`, in every 2D, 3D, mix, container and
  automatic-policy augmentation, in `forward` and `inverse`, and with mask, box, keypoint and class inputs. Values are
  unchanged bit for bit, and compiled graphs keep their copies. The synchronizations that remain are the ones whose
  result the host needs: `inverse()` and the 3D warps invert the transformation matrix with `torch.linalg.inv`, which
  raises for a singular matrix; `RandomThinPlateSpline` solves its spline system with `torch.linalg.solve` for the same
  reason; box inputs are validated (now in one host read instead of three); and
  `RandomTransplantation` and `AdaptiveDiscriminatorAugmentation` draw from data on the device. Parameters as large as
  the image (noise, plasma and illumination maps, elastic noise, salt-and-pepper masks) still cost the host the time of a
  pageable copy, and an augmentation moved to the GPU with `.to("cuda")` samples its parameters on the GPU and still
  reads them back. ([#6084](https://github.com/kornia/kornia/pull/6084))
- `BoxFiltering.to_onnx` without `filter_as_zero` now exports a fixed batch: the batch of `input_shape`, else of `pseudo_shape`, else one image. With the TorchScript exporter (the default before torch 2.9), the graph used to declare a dynamic batch but return only the traced batch's per-image outputs: it ran correctly only at a batch of 5, failed in ONNX Runtime at smaller batches and silently dropped images at larger ones. The dynamo exporter failed. Pass `input_shape=[5, -1, 6]` to keep exporting for 5 images. An `input_shape` with a dynamic (`-1`) batch now raises a `ValueError` that points to `filter_as_zero=True`, which exports a batch of any size as one `(B, D, 6)` tensor. ([#6086](https://github.com/kornia/kornia/pull/6086))
- `kornia.geometry.bbox.bbox_generator3d` now places each far corner at `start + size - 1`, so `infer_bbox_shape3d` reads back the requested width, height and depth and `bbox_to_mask3d` covers `width * height * depth` voxels, as with `bbox_generator` and `Boxes3D.from_tensor(..., mode="xyzwhd")`. Before, the far corner was at `start + size`, and every box measured one voxel larger than requested on each axis. Code that passed `size - 1` to work around this now gets boxes one voxel smaller on each axis and should pass `size`. `CropGenerator3D` did this and now passes `size`, so `RandomCrop3D` and the generator's `src` and `dst` boxes are unchanged. ([#6088](https://github.com/kornia/kornia/pull/6088))
- `rgb_to_hls` and `RgbToHls` now subtract color channels before dividing by chroma, avoiding NaN float16 hue gradients for near-gray pixels such as normalized RGB `(250, 251, 250)`. Hue values also move by rounding-level amounts in every dtype, closer to a float64 evaluation. ([#6089](https://github.com/kornia/kornia/pull/6089))
- `kornia.geometry.camera.dx_distort_points_affine` returns one Jacobian per point: its batch shape is the broadcast of the points' and the params' leading shapes, as documented and as `distort_points_affine` and `dx_distort_points_kannala_brandt` give. `N` points with one set of params returned a single `(2, 2)` Jacobian, and `(B, N, 2)` points with `(B, 1, 4)` params returned `(B, 1, 2, 2)`. A params batch that does not broadcast against the points now raises, where it was ignored. ([#6091](https://github.com/kornia/kornia/pull/6091))
- On torch 2.9.1, `torch.onnx.export(..., dynamo=True)` exports `pyrdown`, `build_pyramid` and `gaussian_blur2d` with a dynamic batch again. They used to fail with "Constraints violated (batch)": the batch check of `filter2d` and of `gaussian_blur2d`'s `sigma` compared a kernel or `sigma` batch of 1 with the input batch before it accepted the 1, and the size-oblivious `torch.export` that this exporter runs then guarded the batch against 1. `KORNIA_CHECK_SAME_SIZE` with `allow_one=True` now accepts a size of 1 before comparing it with the other size. ([#6094](https://github.com/kornia/kornia/pull/6094))
- `camtoworld_to_worldtocam_Rt`, `worldtocam_to_camtoworld_Rt`, `camtoworld_graphics_to_vision_4x4` and `camtoworld_vision_to_graphics_4x4` now return a result in the input's dtype for `uint8`, `int8`, `int16`, `int32` and `int64` input on CUDA, as they already did on the CPU and MPS. CUDA has no integer matrix multiplication, so these functions used to raise `NotImplementedError: "baddbmm_cuda" not implemented for 'Long'` (or the matching error for another integer dtype) there. ([#6096](https://github.com/kornia/kornia/pull/6096))
- `jpeg_codec_differentiable` and `JPEGCodecDifferentiable` now compute `float16` and `bfloat16` images in `float32`,
  quality and quantization tables included, and return the input dtype. The codec ran in the input dtype before, where
  pixels scaled to [0, 255], DCT coefficients of about 1000 and a quality scale of 2500 do not fit an 8-bit or 11-bit
  significand: on random 64x64 images, outputs were 0.15 to 0.25 (`bfloat16`) and 0.11 to 0.18 (`float16`) away from the
  `float64` codec, and image gradients were off by about their own magnitude (relative L2 error 0.5 to 0.9). The output
  is now the `float32` codec's output rounded once to the input dtype. `JPEGCodecDifferentiable` also no longer rounds
  its `float32` tables to the image dtype before passing them to the codec. Inside `torch.autocast` the output now keeps
  the input dtype too: a `float32` image under CPU `bfloat16` autocast used to come back as `bfloat16`. ([#6100](https://github.com/kornia/kornia/pull/6100))
- `find_homography_lines_dlt`, `find_homography_lines_dlt_iterated` and `find_homography_dlt(..., solver="svd")`
  return NaN for a batch element whose DLT system is not finite, such as one with a NaN or infinite point, endpoint or
  weight, and the usual homography for the rest of the batch, on CPU and CUDA, eager and compiled. On CPU, one such
  element made `torch.linalg.svd` raise for the whole batch; the solvers then warned "SVD did not converge" and returned
  uninitialized memory from `torch.empty`. Every homography of the batch was then garbage that changed from call to call
  and had no gradient. Compiled and exported programs raised `LinAlgError` instead. On CUDA, a non-finite weight could
  give a finite but meaningless homography. If the SVD fails on a finite system, the whole batch is now NaN, with the
  same warning. ([#6103](https://github.com/kornia/kornia/pull/6103))
- `focal_loss` and `FocalLoss` raise a `RuntimeError` for a label outside the classes under `torch.compile` on the CPU, as
  they do in eager mode. With `reduction='mean'` or `'sum'` and no `weight`, the loss gathers the target class, and
  Inductor's CPU gather checks its indices inside an OpenMP parallel region: a label past the classes terminated the whole
  Python process (exit code 134, "index out of bounds") on torch 2.5.1, 2.9.1 and 2.14, a negative label did the same on
  2.14, and 2.5.1 and 2.9.1 returned a loss for a negative label when no gradient was needed. The labels are now checked
  with `torch._assert_async` before the gather, and on torch 2.14 a compiled gather reads them clamped into range, so it
  stays in bounds until the check raises. Inductor before torch 2.9 compiles the assert to nothing, so there a compiled
  call still fails as before. The check uses `one_hot`'s messages: an eager CPU call now raises "Class values must be
  non-negative." or "Class values must be smaller than num_classes." where the gather raised "index ... is out of bounds
  for dimension 1 with size C", as the `'none'` and weighted losses already did. CUDA is unchanged: an out-of-range label
  still fails the gather's device-side assert there. ([#6104](https://github.com/kornia/kornia/pull/6104))
- `PinholeCamerasList.clone()`, `scale()` and `scale_()` now preserve camera-list dimensions and apply per-batch scale factors across cameras. ([#6106](https://github.com/kornia/kornia/pull/6106))
- `TinyViT.from_config("21m", ...)` with the `'in1k_384'` or `'in1k_512'` checkpoint (also reached by `pretrained=True` at `img_size` 384 or more) builds with `drop_path_rate=0.1`, the stochastic depth rate those checkpoints were fine-tuned with (Cream `tiny_vit_21m_384` and `tiny_vit_21m_512`). It used TinyViT-21M's 224-px rate of 0.2. Only training changes: DropPath does nothing in evaluation mode. A `drop_path_rate` argument still overrides the default. ([#6107](https://github.com/kornia/kornia/pull/6107))
- `jpeg_codec_differentiable` and `JPEGCodecDifferentiable`, and so `RandomJPEG`, run with autocast disabled, so a
  caller's `torch.autocast` region no longer changes their output, or their gradients when the backward pass runs after
  the region (on CUDA, up to the run-to-run spread of its atomic accumulation unless deterministic algorithms are
  enabled). Autocast ran the codec's matrix products and convolutions in its lower dtype, which rounds pixels scaled to
  [0, 255], DCT coefficients and the quality scale too coarsely: on random 64x64 `float32` images at qualities 99, 90, 50
  and 10, outputs were up to 0.109 (CPU `bfloat16` autocast), 0.014 (MPS `float16`) and 0.088 (MPS `bfloat16`) away from
  the `float64` codec on an Apple M1, and up to 0.03 (`float16`) and 0.12 (`bfloat16`) under CUDA autocast. They now
  equal the call outside autocast, 8.4e-7 from the `float64` codec on the CPU and MPS and 6.9e-7 on CUDA. A backward pass
  inside the region, which `torch.compile` assumes by default, still computes part of the gradient in the autocast dtype. ([#6109](https://github.com/kornia/kornia/pull/6109))
- `sampson_epipolar_distance`, `symmetrical_epipolar_distance`, `left_to_right_epipolar_distance` and `right_to_left_epipolar_distance` accept a single `(3, 3)` fundamental matrix, as their `(*, 3, 3)` shapes document, and return the distances of the batched call without its batch axis. They raised `ValueError: Fm must be a (*, 3, 3) tensor. Got torch.Size([3, 3])`, except the first two on CUDA with fewer than 10,000 points, which accepted it. An `Fm` whose last two dimensions are not `(3, 3)` now raises that `ValueError` on every path. On CUDA the first two raised a `RuntimeError` or `IndexError` from the matrix product instead, and returned values for a `(4, 4)` `Fm` with 4-component points; with one point set and several matrices on the CPU, `sampson_epipolar_distance` raised a `RuntimeError`, or reshaped an `Fm` such as `(2, 9, 1)` into `3 x 3` matrices. ([#6110](https://github.com/kornia/kornia/pull/6110))
- `get_hanning_kernel1d` and `get_hanning_kernel2d` export with torch 2.14's default (dynamo) ONNX exporter again. Built without a `dtype`, the window multiplied an int64 ramp by a float, an `aten.mul.Scalar` that the exporter has no translation for with onnxscript 0.7.1 (`ConversionError`). The ramp is now converted to the default dtype first, the conversion the multiplication made implicitly on the CPU. Eager outputs are unchanged bit for bit with an explicit `dtype`, with a float32 or float64 default dtype, and on the CPU with any default dtype. With a float16 or bfloat16 default dtype on MPS, windows from 2057 (float16) or 259 (bfloat16) taps now round the ramp to that dtype as the CPU does, and differ from before by up to about 16 units in the last place. ([#6129](https://github.com/kornia/kornia/pull/6129))
- - `to_onnx` no longer overwrites the output cache that `show()` and `save()` read. On the TorchScript exporter (the `to_onnx` default below torch 2.9, or `dynamo=False`), the export's run on its random dummy input used to leave that output in the exported module's cache and in the cache of every image module inside it, `AugmentationSequential` included, so `show()` rendered noise after an export. Every cache now holds the same object as before the export, whether the export succeeds or raises.
  - `HFONNXComunnityModel.to_onnx` returns a copy carrying that call's metadata. It used to add the metadata to the stored graph, which is the `ModelProto` the model was built from, so the keys of one call appeared in the caller's graph and in every later export.
  - `HFONNXComunnityModel.resize_back` with a `target_images` that is not a tensor raises a `TypeError` naming the type; it raised a `RuntimeError` without a message.

  ([#6130](https://github.com/kornia/kornia/pull/6130))
- Super-resolution weight loading in `kornia.contrib` and `kornia.models`:

  - The pretrained Real-ESRGAN weights of `RRDBNetBuilder` (also reached through `SuperResolution.from_config`) and the `SmallSRNet` weights behind `SmallSRBuilder` are cached in the torch hub directory through `kornia.core.download.load_state_dict_from_url`, as kornia's other checkpoints are. They were cached under `kornia_config.hub_onnx_dir` (`.kornia_hub/onnx_models`, relative to the working directory), so each new working directory downloaded them again; existing users download them once more. A copy kept for offline use goes to `<torch.hub.get_dir()>/checkpoints/` under the URL's file name: `small_sr.pth` becomes `superres_epoch100-44c6958e.pth`, and the Real-ESRGAN files keep their names.
  - A Real-ESRGAN checkpoint without `params_ema` loads its `params`, the fallback of Real-ESRGAN's own loader; it raised `KeyError: 'params_ema'`.
  - `SmallSRNet.load_from_file` reads a local checkpoint file, as its docstring said; a path raised `ValueError: URL must start with 'http:' or 'https:'`. A path that names no file raises `FileNotFoundError`.

  ([#6131](https://github.com/kornia/kornia/pull/6131))
- `PatchSequential` with `random_apply` now permits a mix augmentation such as `RandomMixUpV2` to be selected, at most once per forward call. Previously, mix members were never selected. A `RandomTransplantation` member, which needs a mask that `PatchSequential` does not take, now raises its missing-mask error when it is selected, as it already did without `random_apply`. ([#6132](https://github.com/kornia/kornia/pull/6132))
- `AugmentationSequential`, `ImageSequential` and `VideoSequential` now track the size that `PadTo`, and `RandomMosaic` with an `output_size`, return. Every later child used to draw its parameters for the canvas before them, so the pipeline differed from applying the children one at a time. An integer-size `Resize`, `LongestMaxSize` or `SmallestMaxSize` returned the wrong size: `PadTo((40, 40))` then `Resize(16)` gave a `(16, 26)` image instead of `(16, 16)`. A resample-mode crop, `RandomCrop`, `RandomResizedCrop`, `RandomAffine`, `RandomPerspective` and `RandomErasing` used the old size for their windows, centres and corners, `RandomElasticTransform` raised a size-mismatch `RuntimeError`, and a later `Resize`, slice-mode `CenterCrop` or `RandomHorizontalFlip` moved boxes and keypoints for the old canvas. The resample-mode crop was also wrong under `torch.export`. ([#6133](https://github.com/kornia/kornia/pull/6133))
- `kornia.onnx` composition and loading:

  - `ONNXSequential`, and `HFONNXComunnityModel` / `DepthEstimation` built on it, chain models whose metadata differ for
    a key: a graph exported by another kornia release (its `version`), exports given different `additional_metadata`
    values, or an onnx-community model whose `source` differs from the kornia pre-processor's. They used to raise
    `ValueError: Can't merge models with different values for the same model metadata property`. The combined model
    holds one entry per key, at its first position, with the value of the last model that sets it; the models passed in
    are not modified.
  - `ONNXSequential` applies `target_ir_version` and `target_opset_version` without `auto_ir_version_conversion=True`,
    as `ONNXModule` does; it used to ignore them silently. Without the flag, only the versions given are changed, and a
    model that already has them is not converted again (`ONNXModule` as well).
  - A conversion to a target version, by `ONNXModule` or `ONNXSequential` with or without the flag, accepts a model that
    lists a metadata key twice, as kornia 0.9.0rc1 wrote after `add_metadata`, and keeps one entry per key with its last
    value, the one ONNX Runtime reads; it used to raise `ValidationError: Your model has duplicate keys in
    metadata_props`. A combined model lists each operator-set domain once, so it can be converted and chained again; the
    conversion used to update only one of the repeated entries, and the next merge raised `Can't merge two models with
    different operator set ids`.
  - `ONNXSequential` warns (`UserWarning`) when the tensor element types of an output and the input it feeds differ at
    a joint: the merged graph does not check a later model's declared input types, so such a chain runs on the earlier
    model's type, or fails later with ONNX Runtime's `Type Error`, as before. Declared dimensions are not compared.
  - A malformed `io_maps` raises a `ValueCheckError` (a `ValueError`) that says `io_maps` holds one list of
    `(output, input)` name pairs per joint, where it used to raise `IndexError` (an entry missing) or
    `ValueError: Output K00-m is not present in g1` (a flat list of pairs, the form the ONNX guide showed). A pair with
    more than two names, whose extra names were ignored, now raises it too. An entry may still be any iterable of pairs
    (a `zip`, `dict.items()`, a generator, a numpy array) and `io_maps` anything indexed by the joint, and entries past
    the last joint are still not read. The `io_maps` annotations of `ONNXSequential`, `ONNXSequential.combine` and
    `HFONNXComunnityModel` are now `list[list[tuple[str, str]]]`, and the guide's example is a list of lists.
  - `ONNXModule`, `ONNXSequential`, `kornia.onnx.load`, `ONNXLoader.load_model` and `ONNXLoader.load_config` take
    file paths as `os.PathLike` objects such as `pathlib.Path` (always a local file), and expand a leading `~` in a
    local path. A path object used to raise `ValueError: Invalid argument type` (`AttributeError` from the two loaders).
    An argument of another type now raises a `TypeError` that names what is accepted, instead of `ValueError`
    (`AttributeError` from the two loaders); for `bytes`, which ONNX Runtime reads as a serialized model, it says to pass
    `onnx.load_model_from_string(data)`.
  - Calling `ONNXModule` or `ONNXSequential` takes CPU torch tensors of a dtype numpy has (not bfloat16 or the float8
    types) as well as numpy arrays, and still returns numpy arrays; a tensor used to raise ONNX Runtime's `RuntimeError:
    Input must be a list of dictionaries or a single numpy array`. A tensor on another device raises a `ValueError` that
    says to move it to the CPU, and a sparse tensor or one of a dtype numpy lacks a `TypeError` that names the input and
    how to convert it.
  - `ONNXLoader.load_config` reads an existing local JSON file, which it used to report as not found ("You may set
    `download=True`"). An argument that is neither an HTTP(S) URL nor an existing file (a directory, for one) raises a
    `RuntimeError` that says so.
  - The `ONNXModule` docstring documents its parameter `op`, one model, and `io_name_conversion` documents that it
    applies any mapping in place, returns the same model, and leaves initializers, `value_info` and subgraphs alone.

  ([#6134](https://github.com/kornia/kornia/pull/6134))
- ONNX exports of `EdgeDetector`, MobileSAM and RT-DETR declare and return what they compute:

  - `EdgeDetector.to_onnx` with the pre- and post-processor exports a graph whose edge maps have the input's size, as
    eager `forward` returns them. The graph returned the maps at the pre-processor's size (352 x 352 for the default
    builder) whatever the input, and only warned that `ResizePostProcessor` is not exported. With a `ResizePostProcessor`
    in `"bilinear"`, `"bicubic"`, `"nearest"` or `"nearest-exact"` mode it now returns one `(B, 1, H, W)` tensor, resized
    with that mode: the `torch.cat` of the eager per-image list. A direct `torch.onnx.export(detector, ...)` gives the
    same graph on the TorchScript exporter and, from torch 2.9, on the dynamo exporter. `image_size` is documented as the
    graph's input size, which the maps have too: the pre-processor keeps the size it was built with. Eager `forward` and
    `ResizePostProcessor` are unchanged, and a `ResizePostProcessor` subclass that overrides `forward` is exported as its
    `forward` computes.
  - `Sam.to_onnx` of a MobileSAM (TinyViT image encoder) on the TorchScript exporter, the default before torch 2.9 or
    `dynamo=False`, traces the encoder at the batch of `pseudo_shape` (1 by default) and declares that batch static, so
    the graph states the one batch it runs at. It declared a dynamic batch, but ran only at the traced batch. Other
    encoders, and the dynamo exporter, keep a dynamic batch.
  - `ObjectDetector.to_onnx(include_pre_and_post_processor=False)` names both outputs of an RT-DETR model, `pred_logits`
    and `pred_boxes` (now declared once as `RTDETR.ONNX_OUTPUT_NAMES`), each with the input's dynamic batch, and keeps the
    detector's dynamic height and width for `image_size=None`. Only the logits were named (`output`), and the boxes came
    out under an exporter-generated name, such as `2814` from the TorchScript exporter. A caller's `output_names` take
    their place, with the same batch; a model that declares no output names keeps the single `output`.
    `RTDETR.to_onnx` given its own `output_names` declares the input's dynamic batch on them too; it declared it on
    `pred_logits` and `pred_boxes`, which the graph then did not have.

  ([#6135](https://github.com/kornia/kornia/pull/6135))
- `Hyperplane.signed_distance`, `abs_distance` and `projection` now preserve point gradients for plain tensor inputs, matching their behavior for `Vector3` inputs. ([#6136](https://github.com/kornia/kornia/pull/6136))
- `depth_to_normals` now widens float16 and bfloat16 inputs to float32 and runs its gradient convolution outside autocast. float16 depth maps gave NaN normals for near surfaces (closer than about 0.25 m at a 1000-pixel focal length) and inside zero-depth holes, normals off by tens of degrees at 0.5 to 1 m, and non-finite gradients; bfloat16 depth maps wider than 256 pixels gave zero normals; float32 depth maps inside an autocast region gave normals off by a median of 1 to 40 degrees (and, under float16 autocast, non-finite gradients); and float16 or bfloat16 depth maps inside a CPU autocast region of the other half dtype raised `RuntimeError`. The output dtype inside autocast now follows the inputs, as it does outside: under CPU autocast, float32 and integer inputs used to return the autocast dtype, and under CUDA autocast, half-precision inputs used to return float32. ([#6137](https://github.com/kornia/kornia/pull/6137))
- Fixed `VideoSequential` raising an `IndexError` for `RandomCutMixV2` with the default `same_on_frame=True`, and with `same_on_frame=False` when the CutMix has `same_on_batch=True` and the batch holds more than one video. With `same_on_frame=True` the mix parameters are now repeated over the frames along the video axis, so every frame of a video cuts the same box from the matching frame of the same partner video. ([#6138](https://github.com/kornia/kornia/pull/6138))
- `LoFTR` in float16 or bfloat16 no longer raises a dtype-mismatch `RuntimeError` on images taller or wider than 2048 pixels: the positional-encoding cache it grows for feature maps above 256 cells now keeps the model's dtype instead of becoming float32. ([#6141](https://github.com/kornia/kornia/pull/6141))
- `decompose_essential_matrix`, `decompose_essential_matrix_no_svd`, `motion_from_essential`, `motion_from_essential_choose_solution` and `project_to_essential` run with autocast disabled, as the solvers of #6072 do, so a caller's `torch.autocast` region no longer changes their result. Under bfloat16 autocast a float32 essential matrix was decomposed with its matrix products in bfloat16: the rotations came back in bfloat16, about `1.6e-3` off and no longer orthogonal, next to a float32 translation, and `motion_from_essential_choose_solution` triangulated its points with bfloat16 projection matrices, which moved them by up to `0.2` at depths of 5 to 9. `motion_from_essential_choose_solution` now casts `K1` and `K2` to the dtype of `E_mat`, so half-precision intrinsics with a float32 matrix, which autocast used to cast to one dtype, still run, and run outside autocast too, where they raised. Any other call made outside autocast is unchanged, and a TorchScript-compiled call keeps the products in the caller's precision as before. ([#6147](https://github.com/kornia/kornia/pull/6147))
- `KORNIA_CHECK_TYPE` now names an expected union by its members, such as `expected int | str`, on every Python version. Its error message said `Union` for `typing.Union[...]`, `Optional` for `typing.Optional[...]`, and on Python 3.14 also `Union` for `int | str`. ([#6149](https://github.com/kornia/kornia/pull/6149))
- `kornia.morphology.dilation`, `erosion` and the operations built on them (`opening`, `closing`, `gradient`,
  `top_hat`, `bottom_hat`) give the same result for a bool or integer `structuring_element` as for the equal float
  one, on every engine. In 0.9.0rc1 a `uint8` or `int8` structuring element raised an overflow `RuntimeError`, a
  bool one raised `NotImplementedError` in `erosion` and gave wrong values in `dilation` when `kernel` excluded a
  cell, and every integer or bool one raised under `engine="convolution"`. The same functions also accept a
  non-contiguous `kernel`, such as a transposed view, under `engine="convolution"`, which used to raise
  `RuntimeError: view size is not compatible with input tensor's size and stride`. Other calls are not affected. ([#6150](https://github.com/kornia/kornia/pull/6150))
- `transform_points` no longer truncates integer operands. Integer points, such as pixel indices from `torch.nonzero`, used to come back in their integer dtype, truncated whenever the transformed coordinate was fractional; they now return in the floating dtype of the transform, or in the default floating dtype when the transform is integer too. An integer transform used to truncate floating points before the product; it is now applied in the points' dtype. `transform_bbox` and `Keypoints.transform_keypoints` change with it. ([#6152](https://github.com/kornia/kornia/pull/6152))
- Fixed `VideoSequential` mixing `RandomMixUpV2`, `PatchMix` and `RandomMosaic` with the wrong frames. With `same_on_frame=True` the partner indices drawn per video now point at frame `t` of the partner video, and with `same_on_batch=True`, `RandomMixUpV2` and `PatchMix` draw one sample per video instead of mixing every frame with the first one. With `same_on_frame=False` and `same_on_batch=True` those two are now drawn for all `B * T` frames, instead of once per frame index with every partner taken from the first video. ([#6154](https://github.com/kornia/kornia/pull/6154))
- `RandomCrop3D` and `CropGenerator3D` now reject a fixed crop size larger than the input volume for empty batches, consistently with non-empty batches. Crops that fit after padding remain valid. A fixed size is now checked without reading tensor data, so on torch 2.14 `torch.compile(..., fullgraph=True)` of `CropGenerator3D`, and of `RandomCrop3D` with `p=1.0`, no longer fails with a data-dependent branch. ([#6156](https://github.com/kornia/kornia/pull/6156))
- `MS_SSIMLoss` now requires `img1` and `img2` to be 4-D `(B, C, H, W)` tensors with the same channels, height and width and equal batch sizes, or a batch size of 1 that is compared with every image of the other batch as before, and raises `ValueError` otherwise (a `ShapeError` for a shape mismatch), also with Kornia checks disabled (#6157). It used to compare only the number of dimensions: a height or width of 1 was broadcast and returned a wrong loss, and so did some 3-D `(C, H, W)` pairs. Inputs of different ranks already raised `ValueError`; every other mismatch or input that is not 4-D ended in a torch `RuntimeError` or `IndexError` and now raises `ValueError`. ([#6158](https://github.com/kornia/kornia/pull/6158))
- The torch 2.14 default (dynamo) ONNX exporter now exports `BoxFiltering` and the detectors that contain it. With `dynamo=True` (the default from torch 2.9), a `BoxFiltering` without a `confidence_threshold` exported a graph that filtered at `0.1` wherever the export succeeded (torch 2.9.1, for example); it now filters at `0.0`, as eager does. ([#6159](https://github.com/kornia/kornia/pull/6159))
- `ObjectDetector.to_onnx` and `BoxFiltering.to_onnx` raise a `ValueError` when a `dynamic_axes` or `dynamic_shapes` argument makes the batch of a per-image-list export dynamic. Such an export has one graph output per traced image, so it used to succeed, declare a dynamic batch, and give wrong results at any other batch size: a detector returned one image's detections for a batch of 3, and a `BoxFiltering` traced at a batch of 5 returned 5 outputs at a batch of 7 and failed in ONNX Runtime at batches of 1 and 3. The check reads the exported graph, so it covers every form of the two arguments, and an override that keeps the batch fixed, such as `torch.export.Dim.STATIC()` or a dynamic detection count only, still exports. Nothing is saved when it raises. `filter_as_zero=True`, and an RT-DETR detector built with `confidence_filtering=False`, export a dynamic batch as before. ([#6162](https://github.com/kornia/kornia/pull/6162))
- `to_onnx` keeps the parameters and transformation matrices that augmentations and their containers stored from their last call, so `inverse()` and `transform_matrix` after an export still describe that call. The export's run on its dummy input used to replace them: after a legacy TorchScript export (the default before torch 2.9, `dynamo=False` from it), `inverse()` raised or inverted the dummy's transform. They now come back as the same objects with the same items, also when the export fails, as the `show()` / `save()` output cache now does (#6130). The export also gives back the global random state it drew from, so the draws after it are the ones they would be without it; it removes the global buffer-registration hook that torch 2.9's `torch.export` leaves after a failed trace, which made every later `torch.compile(..., fullgraph=True)` that registers a buffer raise `Unsupported`; and `RandomResizedCrop` no longer stores each call's crop sizes on its generator, where a torch 2.9 dynamo export left a fake tensor that made `deepcopy` and `pickle` raise. ([#6163](https://github.com/kornia/kornia/pull/6163))
- `solve_pnp_dlt` and `get_perspective_transform` run with autocast disabled, as the solvers of #6072 do, so a caller's `torch.autocast` region no longer changes their result. Under bfloat16 autocast, `solve_pnp_dlt` raised `NotImplementedError: "lu_cpu" not implemented for 'BFloat16'` on the CPU for float32 inputs, and `get_perspective_transform` ran its closed form in bfloat16: the homography between the corners of a 640 x 480 image and four targets mapped those corners 1.36 px off. A call made outside autocast is unchanged, and a TorchScript-compiled `get_perspective_transform` keeps the caller's precision as before. ([#6169](https://github.com/kornia/kornia/pull/6169))
- `rgb_to_yuv422` and `RgbToYuv422` export correctly with the legacy (TorchScript) ONNX exporter on torch 2.9.1 and
  2.14. Before, the exported graph returned the UV planes with the window and window-count axes swapped, shape
  `(*, 2, H, 2)` instead of `(*, 2, H, W/2)`. On torch 2.5.1 that export still raises, as it did before. ([#6171](https://github.com/kornia/kornia/pull/6171))
- `ColorMap` uses the default floating dtype for custom palettes with integer RGB values when `dtype` is omitted, instead of failing during interpolation. ([#6172](https://github.com/kornia/kornia/pull/6172))
- `KRt_from_projection` and `triangulate_points` run with autocast disabled, as the solvers of #6072 do, so a caller's `torch.autocast` region no longer changes their result. Under float16 or bfloat16 autocast, for float32 inputs, `KRt_from_projection` returned `K` and `R` in the autocast dtype on the CPU, with `R` about 3e-3 from orthogonal in bfloat16, and raised `RuntimeError: geqrf: MPS currently supports float32 only` on MPS; `triangulate_points` with its default `solver="eigh"` raised `NotImplementedError` on MPS. A call made outside autocast is unchanged, and a TorchScript-compiled `KRt_from_projection` keeps the caller's precision as before. ([#6175](https://github.com/kornia/kornia/pull/6175))
- `get_closest_point_on_epipolar_line` runs with autocast disabled, as the solvers of #6072 do, so a caller's `torch.autocast` region no longer changes its result. Under CPU autocast it computed the epipolar lines in half precision and raised `RuntimeError: expected scalar type Float but found BFloat16` (or `Half`) for float32 points and `Fm`; under MPS autocast it returned float32 points several pixels off. `Fm` and `pts2` now take the dtype of `pts1`, as autocast used to arrange for half-precision operands: an `Fm` or `pts2` in another dtype than `pts1`, which raised outside autocast (on MPS a mixed `pts2` returned wrong values instead), now gives the result in the dtype of `pts1`. A call with all three in one dtype outside autocast is unchanged, and a TorchScript-compiled call keeps working as before. ([#6179](https://github.com/kornia/kornia/pull/6179))
- `unsharp_mask` and `UnsharpMask` run under `torch.autocast`. Under autocast the blur, a convolution, came back in the autocast dtype, and `torch.lerp`, which does not promote, raised `RuntimeError: expected dtype ... for 'end' but got dtype float` on the mismatch with a float32 image. The blur is now cast to the image's dtype before the `lerp`, so the result keeps the image's dtype, as `2 * image - blur` would and as `kornia.enhance.sharpness` does; outside autocast the cast is a no-op and every result is unchanged. ([#6183](https://github.com/kornia/kornia/pull/6183))
- `BoxFiltering` with `classes_to_keep` now runs on boxes on another device than the class list, such as after `.to(device)` or `.cuda()`, instead of raising `RuntimeError: Expected all tensors to be on the same device`: `forward` moves the class list to the boxes' device, also one passed to `forward`. A `confidence_threshold` tensor of more than one element no longer raises `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` in the constructor. ([#6184](https://github.com/kornia/kornia/pull/6184))
- `Boxes.to_mask`, `Boxes3D.to_mask`, `kornia.metrics.psnr`, `kornia.losses.psnr_loss` and `kornia.losses.PSNRLoss` export correctly with the legacy (TorchScript) ONNX exporter (`torch.onnx.export(..., dynamo=False)`) on torch 2.5.1, 2.9.1 and 2.14. Before, the two `to_mask` graphs had no input and returned an all-zero mask for any box: the legacy exporter traced the CPU loop and lost its in-place writes. Under the legacy exporter and `torch.jit.trace` they now compare a coordinate grid against the box bounds, as the dynamo exporter already did (and `Boxes.to_mask` on CUDA). The PSNR graphs returned a one-element vector of shape `(1,)` where eager returns a 0-d scalar; they now return the eager shape. `Boxes.to_mask` of an image with a side of 0, and `Boxes3D.to_mask` of such a volume, return an empty mask on the loop path (CPU, and CUDA for `Boxes3D`) as the grid path does, instead of raising `RuntimeError: cannot reshape tensor of 0 elements`. Every other eager result is unchanged. ([#6188](https://github.com/kornia/kornia/pull/6188))
- `RandomMixUpV2` and `RandomCutMixV2` now apply `batch_prob` to the class labels. A `params` dictionary whose `batch_prob` selects only some rows, such as a draw replayed on a subset or hand-built parameters, used to label the unselected rows as mixed with their drawn partner and weight although their images came back unchanged; those rows now get their own class twice with lambda 0, matching the image. ([#6189](https://github.com/kornia/kornia/pull/6189))
- `equalize`, `equalize3d`, `RandomEqualize` and `RandomEqualize3D`, and `equalize_clahe` and `RandomClahe` on their
  default (fast) path, now count each pixel in the lookup-table entry it reads: its 8-bit level `(x * 255).long()` in
  the input dtype. Before, they counted it in `floor(256 x)`, the `torch.histc` placement, which put every value in
  `[k/256, k/255)` one bin above the entry it reads; that is about half of a continuous image (#6181). `equalize` of a
  float32 or float64 image is now `PIL.ImageOps.equalize` of those levels, as long as a float32 channel holds at most
  `2**24` pixels (4096 x 4096). A channel whose lookup step is 0 is still returned unchanged.

  On continuous float32 input, over 20 random images per size:

  - `equalize` changes 48–49% of the pixels:
    - at 64 x 64, 42% move by 1 gray level and 1.8% by 2;
    - at 300 x 300, 49% move by 1 and 0.04% by 2;
    - in one 64 x 64 image the lookup step went from 16 to 15, and its pixels moved by up to 18 levels.
  - `equalize3d` changes 50% of the voxels of 4 x 64 x 64 volumes.
  - `equalize_clahe` with the default arguments changes about 77% of the pixels at 256 x 256 and 296 x 296. The pixels
    that change move by 0.73 gray levels on average, and by at most 3.6.

  8-bit images in float32 and float64 are unchanged. On 8-bit images, half-precision `equalize_clahe` now stays within
  0.2 gray levels of float32 in float16 and within 1.5 in bfloat16 (default arguments, 256 x 256 and 296 x 296).

  ([#6190](https://github.com/kornia/kornia/pull/6190))
- `homography_warp(..., normalized_coordinates=False)` and `HomographyWarper(..., normalized_coordinates=False)`, on
  both the direct and the `precompute_warp_grid` path, now warp by a destination→source **pixel** homography: the
  warped grid of source pixel coordinates is normalized under the call's `align_corners` before `grid_sample`, as
  `warp_perspective` does. Previously the pixel grid reached `grid_sample` unnormalized, so almost every sample fell
  outside the image and an identity homography returned a nearly blank image instead of the input. The result now
  matches `warp_perspective` with the inverse homography; the pixel grid is warped and normalized in at least
  `float32`, also under a half default dtype and in a caller's autocast region, scripted or not, so pixel coordinates
  are not rounded to half precision before normalization; only the normalized grid takes the half dtype, as on the
  default path. Callers that pre-multiplied the homography by `normal_transform_pixel` to work around this must drop
  that factor. The default `normalized_coordinates=True` is unchanged. Fixes #6170. ([#6191](https://github.com/kornia/kornia/pull/6191))
- `ObjectDetector.to_onnx` exports a detector whose post-processor returns a tuple with one tensor per image for a batch of one image, as it does for a list, and raises a `ValueError` when a `dynamic_axes` or `dynamic_shapes` argument makes that batch dynamic. It used to declare a dynamic batch: the TorchScript exporter's graph then returned only the traced images' detections at a larger batch, and the dynamo exporter of torch 2.9 and later failed with an error that did not name the cause. A tuple of batched outputs, such as `(boxes, scores)`, still exports a dynamic batch. ([#6192](https://github.com/kornia/kornia/pull/6192))
- `AugmentationSequential` with `mask`, box or `keypoints` data keys, and the `ManyToManyAugmentationDispather` and
  `ManyToOneAugmentationDispather` built from it, export again with torch 2.9's dynamo ONNX exporter when replayed with
  constant `params` and a `RandomAffine` child. Since #5416 the annotation handlers recompute the child's matrix from the
  caller's params under export, and `RandomAffine` copies `center` from them as-is, which that exporter turns into a
  renamed graph constant and rejects. The handlers now compute from a copy of the params, as the image call does.
  Eager outputs are unchanged. Fixes #6166. ([#6196](https://github.com/kornia/kornia/pull/6196))
- `EdgeDetector.to_onnx` and `SuperResolution.to_onnx` with the pre- and post-processor export a model whose output is a list with one item per image for a batch of one image, as `ObjectDetector.to_onnx` does for per-image detections. A tuple with one item per image, which the output kind's reading tells from a fixed set of batched outputs by its length at two batch sizes, counts as such a list, as for `ObjectDetector` (#6192). Such a list comes from a post-processor that returns one, or, for `SuperResolution`, from the network itself or a pre-processor that returns a list of images, which `forward` runs one at a time. Both used to declare a dynamic batch, and the TorchScript exporter (`dynamo=False`) turned the list into one graph output per traced image: the graph returned two maps for a batch of three images, and two maps, one of them empty, for a single image. From torch 2.9 the dynamo exporter failed on such a model, and now exports it as well. A `dynamic_axes` or `dynamic_shapes` argument that makes the batch of such an export dynamic raises a `ValueError` after the export, before anything is saved. The kind of output is first read from a one-image forward pass outside the export, so a `ResizePostProcessor` subclass whose own `forward` returns a list eagerly is now exported for one image too, even if it returns a tensor under export, and a dynamic override of it raises. `EdgeDetector` with a `ResizePostProcessor` that keeps its own `forward` (under the class's own `forward`, without forward hooks), `SuperResolution` with known tensor-output networks and the builders' pre- and post-processors (with unchanged forwards), and a model that returns one tensor keep the dynamic batch. The check after the export reuses a positive list reading. A tensor reading is checked again at the actual tracing batch, so batch-dependent per-image outputs cannot silently produce a dynamic graph that drops images. ([#6197](https://github.com/kornia/kornia/pull/6197))
- `kornia.enhance.threshold` and `kornia.enhance.Threshold` compile with `torch.compile(..., fullgraph=True)` on
  torch 2.9 and earlier, where they failed: the supported-type check no longer iterates the `ThresholdType` enum
  while tracing, which Dynamo there cannot do. ([#6199](https://github.com/kornia/kornia/pull/6199))
- `lovasz_hinge_loss`, `lovasz_softmax_loss` and their module wrappers raise `ValueError` when the prediction and target batch sizes differ, instead of an `IndexError` or `RuntimeError` from inside the loss. A `lovasz_softmax_loss` prediction batch of one is still scored against every target, as before. ([#6200](https://github.com/kornia/kornia/pull/6200))
- `StableDiffusionDissolving` now reports the unsupported version and lists the supported versions when raising `NotImplementedError`. ([#6202](https://github.com/kornia/kornia/pull/6202))
- `RandomCutMixV2` with `same_on_batch=True` and `num_mix > 1` repeated the first mix's pairing and cut in every mix, while the class output credited that donor once per mix; each mix now draws its own pairing and cut, shared by every row of the batch. Calls with `num_mix=1` or `same_on_batch=False` draw exactly as before; with `same_on_batch=True` and `num_mix > 1`, a seeded pipeline now draws differently after this augmentation. ([#6203](https://github.com/kornia/kornia/pull/6203))
- Improve float16 and bfloat16 accuracy in Lab/Luv conversions, `rgb255_to_normals` visualization, Gaussian kernels and blur, and Canny by computing intermediates in float32 and returning the input dtype. Canny no longer overflows when squaring gradients of common 255-level float16 inputs, and decoding a zero float16 normal remains finite. ([#6212](https://github.com/kornia/kornia/pull/6212))
- `GFTTAffNetHardNet`, `HesAffNetHardNet`, `DoGHardNet`, `DoGAffNetHardNet`, `KeyNetHardNet`, `KeyNetAffNetHardNet` and `LightGlueMatcher` now return in evaluation mode (`training=False`), matching `LocalFeatureMatcher` and the pretrained-model convention; outputs are unchanged because their pretrained networks were already in evaluation mode. ([#6221](https://github.com/kornia/kornia/pull/6221))
- `homography_warp3d(..., normalized_coordinates=False)` now warps by a destination→source **voxel** homography: the
  warped grid of source voxel coordinates is normalized under the call's `align_corners` before `grid_sample`.
  Previously the voxel grid reached `grid_sample` unnormalized, so almost every sample fell outside the volume and an
  identity homography returned a nearly empty volume instead of the input. An identity now returns the input and a
  whole-voxel translation moves the volume by whole voxels under both `align_corners` settings, and at
  `align_corners=True` the result matches `warp_perspective3d` with the inverse homography. The voxel grid is warped
  and normalized in at least `float32`, also under a half default dtype and in a caller's autocast region, scripted or
  not, so voxel coordinates are not rounded to half precision before normalization; only the normalized grid takes the
  half dtype, as on the default path. Callers that pre-multiplied the homography by `normal_transform_pixel3d` to work
  around this must drop that factor. The default `normalized_coordinates=True` is unchanged. Fixes #6187. ([#6224](https://github.com/kornia/kornia/pull/6224))
- `Resize`, `LongestMaxSize` and `SmallestMaxSize` with an int size, `Boxes.from_tensor`, `Boxes.to_mask`, `Boxes3D.from_tensor`, `Boxes3D.to_mask`, `validate_bbox3d` and the 3D box functions that call it (`bbox_to_mask3d`, `infer_bbox_shape3d`, `center_crop3d`, `crop_and_resize3d`), `StereoCamera`, `So2.from_matrix`, `Se2.from_matrix` and `jpeg_codec_differentiable` / `JPEGCodecDifferentiable` now compile with `torch.compile(fullgraph=True)` on torch 2.12 and later (CI checks 2.14). Their data-dependent input checks and host-side code paths now follow any graph capture, as they already did under `torch.export` and under `torch.compile` on earlier torch releases. A compiled call on torch 2.12 and later therefore no longer checks for invalid input, as on earlier releases: boxes with a non-finite coordinate (in `Boxes3D.from_tensor`, the `'xy*'` modes of `Boxes.from_tensor` and the 3D box functions that call `validate_bbox3d`) or a non-positive extent, unequal 3D box edges, a stereo rig with differing intrinsics or a zero or reversed baseline, a rotation block that is not of the form `[[a, -b], [b, a]]`, or a JPEG quality outside `[0, 100]`. `crop_by_boxes3d` skips the same 3D box checks in a compiled call, but it reads its output size from the boxes and still does not compile in one graph. Eager calls still raise. Under capture, `Boxes.to_mask` and `Boxes3D.to_mask` take their vectorized path, which fills the same mask. ([#6226](https://github.com/kornia/kornia/pull/6226))
- The `[0, 1]` input clamp that range-dependent 2D intensity augmentations gained in #5980 is cheaper. A call that
  records no reverse-mode gradient and carries no forward-mode tangent now costs a single `clamp`. A call that needs
  either derivative costs a comparison with the clamped copy and a selection, instead of two bound comparisons, their
  conjunction and a selection. Values and gradients are unchanged, apart from the sign of a zero: on MPS and under
  `torch.compile` a `-0.0` pixel can now come back as `+0.0`, as `clamp` returns it there. On an i7-14700K CPU
  (4 threads, batch 8, 256×256, float32), `RandomBrightness` and `RandomContrast` take about 300 µs per call instead of
  830 µs, against about 240 µs in 0.8.3, which had no input clamp. On an RTX 4090 they are 1.2–1.4× faster. ([#6267](https://github.com/kornia/kornia/pull/6267))
- `ColorJiggle`'s channel order and `RandomMixUpV2`'s pairs are drawn as an int64 permutation, whatever the dtype the
  parameter generator samples in. A generator set to `bfloat16` used to raise `"randperm_out_cuda" not implemented for
  'BFloat16'` on CUDA. That happened through `set_rng_device_and_dtype`, through `.to(torch.bfloat16)` on the
  augmentation, and, since containers and parent modules now pass `.to()` on to their augmentations, through
  `.to(torch.bfloat16)` on any model that contains one. In `float16`, a MixUp batch above 2048 could not be permuted
  exactly. torch draws the same permutation for every dtype, so the parameters of every call that worked before are
  unchanged. ([#6268](https://github.com/kornia/kornia/pull/6268))
- `unproject_points_z1` and `compute_correspond_epilines`, compiled with `torch.jit.script` and run on CUDA, returned
  an empty `int32` tensor for an empty batch (no points) once the scripted function had run on non-empty inputs:
  TorchScript's CUDA fuser miscompiles a concatenation of a fused product when every input is empty. Both now avoid
  that group: `unproject_points_z1` concatenates the depth expanded to the product's shape, and
  `compute_correspond_epilines` scales `F x` directly instead of concatenating the scaled components. Eager results are
  bitwise unchanged. `unproject_points_z1` also accepts a depth whose leading dimensions broadcast against the points,
  where it used to raise from `torch.cat`, and raises `ShapeError` for a depth whose trailing size is not 1, where a
  `(..., 2)` depth used to return a meaningless `(..., 4)` result.

  `kornia.image.image_to_string` and `print_image` match an integer image's pixels at their exact 0–255 values on every
  device, as documented. The image was divided by 255 and multiplied back, which is exact on CPU but moved 126 of the 256
  `uint8` levels off their integer on CUDA, where a division by a scalar is a multiplication by its reciprocal. A CUDA
  image could therefore print some pixels in a different colour than its CPU copy. CPU output is unchanged. ([#6269](https://github.com/kornia/kornia/pull/6269))
- An empty batch through a shape-changing augmentation with `p < 1` now comes back at the configured output size, as
  with `p=1`, instead of at the input size: `RandomCrop((4, 4), p=0.5)` on a `(0, 3, 6, 8)` batch gives
  `(0, 3, 4, 4)`, as in 0.8.3. This covers `RandomCrop`, `CenterCrop`, `RandomResizedCrop`, `Resize`,
  `LongestMaxSize`, `SmallestMaxSize`, `RandomCrop3D` and `CenterCrop3D`, the masks that follow the image, and the
  sizes `AugmentationSequential` records for its later stages and `inverse`. A non-empty batch that the gate skips
  entirely still keeps its input size (#4429). ([#6272](https://github.com/kornia/kornia/pull/6272))
- Video boxes passed through `VideoSequential(RandomMosaic)` now follow the mosaic of their own frame, as image boxes do,
  with `same_on_frame=True` or `False` and with `p < 1`; before, they were returned untransformed. Frames that a partial
  gate leaves with fewer boxes are padded to the mosaic box count with rows that export as zero in every mode, also after
  a later geometric transform, and `VideoBoxes.to_tensor()` returns them as one `(B, T, N, ...)` tensor. ([#6273](https://github.com/kornia/kornia/pull/6273))
- `distance_transform` on MPS now stops once its cascade settles, as on CPU and CUDA: an 8 x 1 x 256 x 256 blob mask takes
  9 ms on an M1 Pro, against 79 ms before and 12 ms in 0.8.3. Reading whether the cascade has settled costs MPS a few
  rounds' work, so it reads less often there; a cascade that cannot stop early, such as one from a single corner source,
  takes 5 % (512 x 512) to 20 % (64 x 64) longer than before. CPU and CUDA are unchanged. ([#6274](https://github.com/kornia/kornia/pull/6274))
- `adjust_brightness` and `adjust_contrast`, and with them `RandomBrightness`, `RandomContrast` and `ColorJiggle`,
  clamp their own freshly computed result in place when no gradient is recorded. That saves one image-sized temporary
  per call. At batch 32 of 3×256×256 float32 on CPU, the extra 25 MB buffer made glibc trim and re-fault the heap on
  every call (18,400 page faults per call against 12,256), and `RandomBrightness` took 8.6–9.7 ms against 6.2 ms in
  0.8.3. It now takes 7.5 ms, the remaining cost being the `[0, 1]` input clamp added in #5980. Values, gradients and
  the caller's tensor are unchanged. ([#6276](https://github.com/kornia/kornia/pull/6276))
- `RandomCutMixV2` and the PatchMix generator draw their Beta mixing coefficient in float32 when their parameter
  generator is set to float16 or bfloat16, as the position samplers already do (#4553). torch draws Beta through
  Dirichlet, which has no CPU half-precision kernel. A CPU generator in half precision used to raise `"dirichlet" not
  implemented for 'Half'` on every call. That happened after `set_rng_device_and_dtype`, after `.to(torch.float16)` on
  the augmentation (in 0.8.3 too), or, since containers and parent modules now pass `.to()` on to their augmentations,
  after `.to(torch.float16)` on any model that holds one. CutMix builds the concentration from `beta` directly in
  float32, so a `beta` outside the float16 range keeps its value. Float32 and float64 draws are unchanged. ([#6277](https://github.com/kornia/kornia/pull/6277))
- `bilateral_blur` and `joint_bilateral_blur` are much faster on MPS: an 8 x 3 x 256 x 256 batch with a 5 x 5 kernel takes
  40 ms on an M1 Pro, against 556 ms before and 30 ms in 0.8.3; a grayscale batch takes 18 ms, against 378 ms before and
  208 ms in 0.8.3; and the RGB forward and backward pass takes 127 ms, against 1.1 s before. The filter's window tensors
  are now laid out in their memory order, which also makes CPU inputs that take the unfold path (small or differentiated
  ones) up to 30 % faster. Without a gradient the unfold path frees each window-sized temporary once used, so its
  peak memory is 27-42 % lower. Values change only by the rounding of a different summation order. ([#6280](https://github.com/kornia/kornia/pull/6280))
- A fresh transfer that `validate` rejects is now removed even when no other source could have used
  the emptied path -- which is every `download_hf_file` caller, since it passes one URL. Those bytes
  arrived during the call and were refused, so unlike the ambiguous load failures
  `load_state_dict_from_url` weighs, keeping them would end the call having added a poisoned entry to
  a cache that had none. An offline re-fetch also reports the validator's rejection rather than the
  network error stacked on top of it, so the message names the file rather than an entry that is
  intact and, by then, restored. (#4367)
- Canny hysteresis preserves the input dtype, avoiding a convolution dtype mismatch for half-precision images. (#4393)
- Constructing `ScaleSpaceDetector` with `compile_modules` no longer permanently mutates the process-global
  `torch._dynamo.config.capture_dynamic_output_shape_ops` setting. (#4134)
- Correct non-zero sensor tilt in `tilt_projection` and `distort_points` to match OpenCV,
  restoring forward/inverse round trips and correcting `undistort_image` sampling. (#4384)
- Corrected stereo disparity validation errors to describe the required channels-last
  `(B, H, W, 1)` layout and report the received shape. (#4380)
- Corrected the matrix-shape checks shared by `PinholeCamera` and `PinholeCamerasList`, the
  `pixel2cam` coordinate-shape check, and the `cam2pixel` coordinate/projection checks. Invalid
  inputs now raise the intended `ValueError` instead of being accepted or failing later in tensor
  operations. The rank-4 matrices used by `PinholeCamerasList` remain supported. (#4387)
- DeDoDe's vendored DINOv2 no longer probes for `xformers`; the pure-PyTorch path it always ran is the
  only one. (#4288)
- Define singleton pixel axes at normalized center, reject non-positive normalization sizes, preserve empty warp
  destinations, and deprecate the now-unused `eps` normalization parameters (#4006). The four
  `{de,}normalize_pixel_coordinates{,3d}` helpers and `normal_transform_pixel{,3d}` now derive their scale from the
  size directly instead of rounding the size into the coordinate dtype first, so results at non-degenerate sizes can
  move toward the exact value: by up to one ulp in `float32`/`float64`, but materially more in `float16`/`bfloat16`,
  where the old code could not represent the size at all (`denormalize_pixel_coordinates` at `bfloat16` and size 3000
  returns 1496 where it returned 1504).

  A singleton axis previously divided by zero (or by the `eps` substituted for it), so the change is not confined to
  those helpers: it reaches every operation that normalizes a coordinate frame with a size-1 dimension.
  `create_meshgrid(1, 4, normalized_coordinates=True)` returned `nan` in the singleton component and now returns `0`;
  `warp_perspective`, `crop_by_transform_mat`, `homography_warp` and `warp_image_tps` onto a 1-pixel-high destination
  returned all-`nan` and now return the finite row the 2-pixel control produces; `spatial_soft_argmax2d` and
  `spatial_expectation2d` on a 1-pixel-high input returned `nan` and now return `0`; and `conv_soft_argmax2d`,
  `conv_soft_argmax3d` and their `ConvSoftArgmax*` module forms shift by one normalized unit when a kernel or input
  axis is `1` (`conv_soft_argmax2d` with a `(3, 1)` kernel returned `[-1.6667, -1.0, ...]` and now returns
  `[-1.0, -0.3333, ...]`).
- Drop the constant homogeneous row from `laf_to_boundary_points`. The function appended a `[0, 0, 1]` row to every
  LAF so that it could divide the result by a homogeneous coordinate that is always exactly `1`; it now multiplies
  by the `(2, 3)` LAF directly, and moves its 3-column basis to the LAF's device and dtype before broadcasting it
  instead of after (the previous order materialized, and on an accelerator transferred, one copy per LAF -- ~12 MiB
  per call at `N = 20000` for a tensor with `n_pts` distinct rows). The win is not the transfer: torch's CPU
  batched gemm falls off a fast path at the third row, so `bmm` on a `(B, 3, 3)` operand costs 23x a `(B, 2, 3)` one
  for 1.5x the arithmetic. Quiet back-to-back A/B against base `8db1499a` via `benchmarks/feature/laf_ops.py`
  (Apple M1, torch 2.14.0, 4 threads, B=1 N=20000): `laf_to_boundary_points` 1.07M -> 29.6M LAFs/s eager on CPU and
  1.77M -> 4.27M on MPS; `laf_is_inside_image`, which calls it with `n_pts=12` on every detector forward,
  9.75M -> 29.0M on CPU and 2.70M -> 8.53M on MPS. Results are bitwise identical on CPU `float16` and `bfloat16`
  and shift by at most 1.42 machine epsilon elsewhere, relative to the largest coordinate of the call -- the two
  gemm shapes round differently. Mean error against a `float64` reference is slightly lower than before, and
  `laf_is_inside_image`'s boolean masks were bitwise identical across a 126-case device/dtype/shape sweep. (#4217)
- Eager bbox validation now rejects NaN and infinite coordinates: `validate_bbox` returns `False` and
  `Boxes.from_tensor(..., validate_boxes=True)` raises `ValueError` for the `xyxy`, `xyxy_plus` and `xywh`
  modes, instead of accepting them as valid geometry (closes #4238). (#4243)
- Fix 2-D, 3-D, and mix augmentations raising a cross-device `RuntimeError` when their random parameters are generated
  on the default CPU device and their input is on MPS or CUDA (#4164). Since #3722, four branchless `torch.where`
  blends used the CPU `batch_prob` mask directly with accelerator outputs or transformation matrices. The selection
  mask now moves to the output device at each blend boundary, preserving CPU random-number generation and the
  ONNX/fullgraph-friendly blend. Calling `augmentation(x_accelerator)` therefore works again without first moving the
  augmentation module or its RNG to the accelerator (#4151).
- Fix `Boxes.merge` concatenating along the vertex axis instead of the box axis
  for unbatched `(N, 4, 2)` input. The old `dim=1` was correct for batched
  `(B, N, 4, 2)` data (where dim 1 is the box axis) but wrong for 3-D tensors
  (where dim 1 is the vertex dimension); `dim=-3` is correct in both cases and
  the behaviour of the batched path is byte-identical. For list-backed inputs,
  each batch row is now repacked with its real boxes before all trailing padding,
  and the combined per-image padding counts are retained. This makes ``to_tensor``
  trim only padding instead of dropping merged boxes or exposing old padding as
  real boxes (#4168, #4175).
- Fix `Boxes3D.from_tensor(..., mode="xyzwhd")` to preserve width, height, and depth field order for asymmetric
  boxes. (#4189)
- Fix `DenseSIFTDescriptor` registering a process-global cached tensor as its `_poolingconv_weight` buffer, so that
  one instance's `load_state_dict` silently corrupted every other instance and every instance constructed later
  (#4068). `_get_reshape_kernel` memoises `torch.eye(numel)` per `numel` and used to return a *view* of the cached
  tensor; `.float()` on an already-float tensor returns the same object, so no copy happened between the cache and
  the buffer, and `load_state_dict` copies into buffers in place. Loading a checkpoint into one `DenseSIFTDescriptor`
  therefore overwrote the shared identity matrix, with no error raised and every descriptor built from it wrong from
  then on. The cache is removed rather than made safe: cloning on the way out is what a safe cache requires, and a
  clone costs more than rebuilding the identity at every size measured (2.6 us against 2.0 us at the default
  `numel = 8 * 4 * 4`; 3.9 ms against 2.8 ms at the former 4096 bound, which also retained 67 MB for the lifetime of
  the process). Descriptor output is byte-identical to before.
- Fix `SIFTDescriptor.gk`, `PatchDominantGradientOrientation.weighting` and `PatchAffineShapeEstimator.weighting`
  being plain attributes, so `nn.Module.to()` left them behind on the original device and dtype (#4069). They are
  now non-persistent buffers: `.to()`, `.cuda()`, `.half()` and friends move them, and because they are fully
  determined by `patch_size` they stay out of `state_dict()`, so existing checkpoints still load with
  `strict=True`. The three `forward` methods also cast into a local instead of rebinding the attribute, so a call
  no longer leaves the module's kernel in whatever dtype and device the last input happened to have. Numerical
  output is unchanged: the kernels were already cast to the input's dtype and device inside `forward`.
- Fix `find_essential` and `run_5point` raising `TypeError: all() received an invalid combination of arguments` on
  PyTorch before 2.2 (#4043). `run_5point`, `_solve_2x2_tikhonov_safe` and the scripted Nister helper each reduced
  over two axes at once with `Tensor.any(dim=(-2, -1))` / `Tensor.all(dim=(-1, -2))`, and multi-dimension `any`/`all`
  only landed in PyTorch 2.2, so the whole five-point solver was unusable on the declared `torch>=2.0` floor. The
  reductions now flatten the two trailing axes first, which is equivalent on every version.
- Fix `laf_is_inside_image` treating the image extent as `(w, h)` rather than `(w - 1, h - 1)`, which made its
  bounds asymmetric: the lower bound rejected anything left of `x = 0`, but the upper bound accepted `x = w`, a
  full pixel past the last valid column `w - 1` (#4064). Valid pixel coordinates run `0 .. w-1` and `0 .. h-1` --
  the convention `get_laf_center` documents and the one `normalize_laf`/`denormalize_laf` already use -- so the
  upper bound is now `w - 1 - border` and `h - 1 - border`. A LAF whose boundary points reach past the last valid
  pixel coordinate is now reported as outside; anything strictly inside is unchanged. The equivalent inlined check
  in `ScaleSpaceDetector._process_octave` moved with it, so detections within one pixel of the right or bottom edge
  that used to survive its `border=5` filter are now discarded. That inlined check also computed its `max |sin|`
  constant with the wrong angular spacing (`2*pi/11` instead of the `2*pi/10` that `laf_to_boundary_points(n_pts=12)`
  actually samples), inflating the tested x-extent by 4% and making the inline check stricter than the reference it
  claims to reproduce; the two now agree exactly. That correction is a loosening in x, so it can change
  `ScaleSpaceDetector` output on its own: a detection whose x-extent falls between the two constants used to be
  discarded and is now kept. It is rare -- the detection has to land in a band of width `0.039 * half_s` against the
  x bound -- but when it fires it can promote the strongest response in the image, so it is not output-neutral.
- Fix `match_fginn`'s geometric consistency check comparing every query's candidates against **query 0**'s
  candidates instead of against the query's own 1st nearest neighbor, which changes every `match_fginn` and
  `GeometryAwareDescriptorMatcher("fginn")` result -- the latter being that class's *default* mode (#4062). The distance
  `kdist[i, k] = || xy2[idx[i, k]] - xy2[idx[0, k]] ||` was measured from a slice of dim 0 that broadcast query 0's
  whole candidate list over the batch; it is now `candidates_xy[:, 0:1]`, i.e.
  `kdist[i, k] = || xy2[idx[i, k]] - xy2[idx[i, 0]] ||`. Three consequences disappear with it. The geometric term
  was inert -- an unrelated query's candidates are almost always farther apart than `spatial_th`, so nothing was
  penalized and the raw 2nd nearest neighbor was used, making `match_fginn` behave as `match_snn`: on a 300x300
  descriptor set with planted near-duplicates, 15 of the 15 matches shared with `match_snn` had an identical ratio
  before and 14 of 15 after. Query 0 always matched, since `candidates_xy[0] - candidates_xy[0]` is identically zero,
  so all of its candidates were penalized and its ratio collapsed to ~0. And a query's ratio depended on the other
  queries in the batch: with only query 0 changed, one query's ratio moved between `0.9005` and `2.8e-08`. The seven
  existing FGINN tests pass unchanged on both sides, which is how this survived; two tests that discriminate it were
  added. `match_fginn`'s docstring now also records the saturation corner, where every candidate falls within
  `spatial_th` of the 1st nearest neighbor, the ratio collapses towards zero and the match is accepted.
- Fix `symmetric_transfer_error` returning `NaN` instead of `max_num` for singular homographies.
  When `safe_inverse_with_mask` flags an input as non-invertible, the unshielded inverse matrix containing
  non-finite values propagated `NaN` through `oneway_transfer_error` and the arithmetic mask
  `(there + back) * mask + max_num * (~mask)`, poisoning both the forward error and autograd gradients.
  Singular rows are now replaced by the identity *before* the differentiable inverse is taken, so the
  gradients with respect to both the correspondences and `H` itself stay finite (zero on the singular
  rows), and the final masking uses `torch.where` to cleanly assign `max_num`. Values for invertible
  homographies are unchanged; for `squared=False` a singular row now scores `max_num` rather than the
  previous `(max_num + eps).sqrt()`, matching the documented contract. (#4203)
- Fix `torch.jit.script` of `rgb_to_yuv`, `yuv_to_rgb`, `rgb_to_xyz` and `xyz_to_rgb` on older PyTorch (#4043). Their
  shared `kornia.color.utils._apply_linear_transformation` helper annotated its optional argument with the PEP 604
  `torch.Tensor | None` form, which the TorchScript compiler on the declared floor does not accept (reproduced on
  PyTorch 2.1.2, while PyTorch 2.9.1 compiles it), so scripting any of the four conversions failed. The annotation is
  `Optional[torch.Tensor]` now, which every supported version accepts.
- Fix patch extraction on MPS returning darkened patches for any LAF that touches an image border (#4063).
  MPS has no `padding_mode="border"` for `torch.nn.functional.grid_sample`, so `extract_patches_simple` and
  `extract_patches_from_pyramid` emulate it with zero padding and a clamped grid. The clamp was `grid.clamp(-1, 1)`,
  but with `align_corners=False` normalized coordinate `±1` is the outer *edge* of the border pixel, not its center,
  so bilinear sampling blended the border pixel with the zero padding and returned roughly half its value. The grid
  is now clamped to the outermost pixel centers, `±(1 - 1/size)` per axis, which reproduces `padding_mode="border"`
  exactly. On a patch overlapping the image corner the maximum deviation from the CPU result drops from `0.395` to
  `2.5e-6` on a `[0, 1]` image, with 64.7% of the patch's pixels previously off by more than `1e-3`. Every descriptor
  built on these patches — `get_laf_descriptors`, `LAFDescriptor`, `SIFTFeature`, `KeyNetAffNetHardNet` and the rest —
  moves with them for keypoints near the border. The CPU and CUDA paths never took this branch and are byte-identical
  to before.
- Fix the empty-result paths in `kornia.feature.integrated` disagreeing with the corresponding non-empty ones
  (#4065). `get_laf_descriptors` returned a hardcoded width of 128 with the *LAF's* dtype and device when it was
  handed no keypoints, so an empty batch through, say, a 256-d descriptor produced a `(B, 0, 128)` result that could
  not be concatenated with a real one; it now runs the descriptor once on a zero patch and returns its actual width
  (the flattened per-patch output, matching the non-empty path's `.view(B, N, -1)`), dtype and device. That is a
  widening of the empty path's failure surface: a third-party descriptor that raises on a zero input now raises here
  where it previously returned quietly. `LocalFeatureMatcher.no_match_output` returned `lafs0`/`lafs1` of shape
  `(0, 0, 2, 3)` where the success path returns `(1, NC, 2, 3)`, so `lafs0[0]` raised `IndexError` exactly when
  nothing matched; the empty tensors now keep the leading batch of 1. That is a shape change on a public return
  value -- code using `out["lafs0"].shape[0] == 0` as its no-match signal now sees `1` and should test
  `out["keypoints0"].shape[0] == 0`, which was already the consistent signal. `LocalFeatureMatcher.forward`'s
  docstring also stops claiming `confidence` is in `[0, 1]`: it is `1 - distance`, which for the raw-distance
  matchers `nn` and `mnn` goes at least down to -1 on unit-norm descriptors, and is unbounded below for arbitrary
  ones.
- Fix the inverted `SigLip2` attention-mask polarity, so every call that passes an `attention_mask` changes (#4043).
  All three mask branches of `SigLip2Attention` (2-D `(B, N)`, 3-D `(B, N, N)` and 4-D `(B, 1, N, N)`) negated the
  mask before handing it to `torch.nn.functional.scaled_dot_product_attention`, whose boolean form reads `True` as
  *attend*, not as *mask out*. Every position the caller asked to attend to was masked out and every padded position
  attended to, so an all-ones mask left every query row fully masked. The symptom is version-dependent because a
  fully masked row returns `0` from SDPA on PyTorch 2.5 and later and `nan` on PyTorch 2.4 and earlier:
  `SigLip2.get_text_features(..., attention_mask=...)` has been returning a zeroed attention output on current
  PyTorch and `nan` text features on the declared `torch>=2.0` floor. 2-D padding masks are now broadcast over the
  key axis only, matching `SiglipTextTransformer.create_bidirectional_mask` in Hugging Face `transformers`, so a
  partially padded sequence no longer produces a fully masked row. A row whose mask is entirely zero — an empty
  caption in a batch — is still fully masked and still follows the SDPA behavior above.
- Fixed `RandomPlasmaBrightness`, `RandomPlasmaContrast`, and `RandomPlasmaShadow` to replay deterministically from stored `params=`. (#4462)
- Fixed `dx_distort_points_kannala_brandt` to return the true Jacobian of `distort_points_kannala_brandt`, including a finite affine Jacobian at the origin. (#4277, #4368)
- Fixed `unproject_points_z1` depth shape handling for singleton and multi-axis batches,
  accepting both trailing-singleton and flat depth tensors. (#4355)
- Fixed fixture teardown between files when running half-precision tests with
  `--isolate-half-precision`, including reporting parent fixture finalizer errors. (#4388)
- Follow-up fixes to the documentation revamp (#4155): the landing page's filtering card quoted the combined
  `filters`/`color`/`enhance`/`morphology` count next to a `kornia.filters` label, the "Why Kornia?" hero carousel
  resumed after a visitor picked a tab, the Community page linked to the now-unregistered `librecv.org`, and the
  GPU page promised CUDA measurements the committed benchmark runs do not yet contain. `pydata-sphinx-theme` is now
  version-bounded, since the site's CSS and JS target theme internals. (#4173)
- Keep `extract_patches_simple` and `extract_patches_from_pyramid` off a broken reduced-precision CPU kernel: for a
  `float16`/`bfloat16` CPU image, the `grid_sample` kernels in torch <= 2.9 read out of bounds when a rotated patch
  straddles the image border and return NaN or garbage values far outside the image's range. Both extractors now
  sample reduced-precision inputs in `float32` on every device and cast the patches back. This fixes the CPU kernel
  bug and avoids normalized-coordinate precision loss on large images; half-precision patches change on every
  backend, with the CUDA cost described under **Breaking changes**. Both extractors also replace their
  per-image Python loop with a folded batched `grid_sample` over a `(B, N*PS, PS, 2)` grid, split along `N` when
  needed to bound its workspace. Both forms compile under `fullgraph=True`, but the loop was unrolled at trace time
  into one `grid_sample` per batch element, so the graph grew with the batch and was recompiled for every new batch
  size: tracing `extract_patches_simple` over batches of 2, 3, 5 and 7 built four graphs of 2/3/5/7 `grid_sample`
  nodes before and builds two graphs with one node each now.
- Made uncompiled `RandomGaussianIllumination` instances serializable with `pickle` and `torch.save`,
  preserving parameter replay and the `compile()` execution path after restoring the module. (#4457)
- Make YUV and XYZ transformations compute integer inputs in `float32` instead of truncating their kernels, and
  preserve directly constructed `float64` coefficients. This makes `rgb_to_yuv(uint8)` return `float32` on the
  input's original 0--255 scale, fixes signed-integer YUV results, and removes the existing float32 coefficient loss
  from float64 XYZ conversions. The two private linear-transformation helpers are now one implementation, whose CPU
  path uses its dtype/device-aligned compute operands (#4053).
- Make `guided_blur` / `GuidedBlur` work in `float16` and `bfloat16` when the guidance has more than one channel.
  Multi-channel guidance solves a per-pixel `C x C` system, and `torch.linalg.solve` has no half-precision LU
  kernel, so every such call failed: on CPU with `NotImplementedError: "lu_cpu" not implemented for 'Half'`, and on
  MPS with a hard `Only MPSDataTypeFloat32 is supported` assertion that aborted the process rather than raising.
  The system is now solved in `float32` and cast back. A tensor `eps` also takes part in dtype promotion, so
  the common `eps=torch.tensor(0.1)` (float32) widened the matrix past the right-hand side and made `solve`
  reject the pair; both operands are now solved in one dtype. Single-channel guidance never reached the solver
  and is unaffected. `float32` and `float64` results are bit-identical to before, and the op still compiles with
  `fullgraph=True` in one graph. This takes `tests/filters/test_guided.py` from 98 failed / 67 passed to
  169 passed under `--dtype=float16,bfloat16`.
- Make `load_pointcloud_ply` and `load_pointcloud_ply_binary` parse the PLY header instead of skipping a fixed
  number of lines. Both loaders skipped `header_size=8` lines and read everything after them as `x y z` triples, so a
  minimal PLY file (seven header lines, no `comment`) lost its first point to a header line -- the binary reader
  returned header bytes decoded as doubles -- and any file with extra vertex properties (normals, colours) or a
  `face` element after the vertices was rejected or misread. The readers now stop at `end_header`, take the point
  count from `element vertex N`, select `x`, `y`, `z` by property name whatever their scalar type, skip trailing
  elements, and read big-endian payloads. `header_size` is deprecated and ignored (a `DeprecationWarning` is
  emitted when it is passed). Files without a proper PLY header, which the old line-skipping loader could read
  by accident, are now rejected with a `ValueError`, and so is a file that declares a `list` property among the
  vertex properties -- a list has no statically known length, so the columns after it cannot be located. The
  binary reader additionally rejects a `list` property in an element that *precedes* the vertices, whose byte
  length it would have to know to reach the vertex data; the ASCII reader reads such a file, since every ASCII
  element instance is one line whatever it contains.
- Make `yuv_to_rgb` the exact inverse of `rgb_to_yuv`, so an RGB → YUV → RGB round trip is now limited only by the
  input dtype instead of losing up to `1.36e-3` (in B, at `rgb = (1, 1, 0)`) at every precision, `float64` included
  (#4044). The inverse kernel was a separately rounded copy of the published BT.470-5 M/PAL inverse relations rather
  than the inverse of the rounded forward kernel kornia actually ships, and one of its literals — `2.029`, where the
  inverse of kornia's forward kernel is `2.03199968` — carried most of the error. `yuv_to_rgb` output therefore moves
  by up to `1.23e-4` in R, `9.13e-4` in G and `1.60e-3` in B over the documented YUV domain, and `yuv420_to_rgb` and
  `yuv422_to_rgb` move with it. The forward direction (`rgb_to_yuv`, `rgb_to_yuv420`, `rgb_to_yuv422`) is unchanged.
  Agreement of the inverse with the standard's own relations improves overall, from `1.54e-3` to `5.24e-4`, though R
  alone moves the other way (`1.54e-4` to `2.77e-4`). The `.. warning::` blocks that documented the defect on the six
  affected functions and classes are gone.
- Make calibration `distort_points` and `undistort_points` tilt checks compatible with
  `torch.compile(fullgraph=True)`, preserving eager behavior. (#4391)
- Non-maxima suppression with a window larger than `(7, 7)` no longer builds a `(k*k, 1, k, k)` one-hot
  convolution to gather each neighbour into its own channel. On CPU in half precision that convolution
  has no vectorized kernel and falls back to `_slow_conv2d_forward`, where it accounted for 99% of a
  `MultiResolutionDetector.detect` call. The window minus its centre is now covered by rectangular
  `max_pool2d` regions whose full-width slabs share one column pass, costing `O(ky + kx)` taps per
  position rather than `ky * kx - 1`, with the centre excluded by construction so the suppression stays
  strict. `detect` on a 240x240 image drops from 5757 ms to 27 ms in float16 and from 73.8 ms to 13.8 ms
  in float32; a 353x353 `k = 15` suppression drops from 3150 ms to 10.5 ms in float16 and from 42.6 ms to
  8.7 ms in float32, and from 12.1 ms to 0.15 ms for 1024x1024 `k = 21` on CUDA. (#4242)
- Preserve empty color tensors on accelerator backends instead of asking `reshape` to infer an ambiguous batch
  dimension. This lets the YUV420 and YUV422 empty-input conversions return their documented empty RGB output on
  MPS, matching CPU behavior (#4185).
- Raise `NotImplementedError` instead of `RuntimeError` for an integral `src` on the empty-`dsize` paths of
  `warp_affine`, `warp_perspective`, `remap`, `warp_affine3d` and `warp_perspective3d`, matching what the non-empty
  path already raises from `grid_sample` (#4031). The degenerate path had its own explicit guard that fired first
  with a different exception type, so the exception a caller saw for the same invalid input depended on whether
  `dsize` had a zero dimension. The message is unchanged. `NotImplementedError` derives from `RuntimeError`, so
  `except RuntimeError` around these functions still catches it; only code matching on the exact type sees a
  difference.
- Reduce local-feature extraction overhead by accumulating orientation histograms directly,
  batching built-in DoG extrema refinement, selecting coordinates before combining response signs,
  and using faster activation layouts in KeyNet and CPU HardNet. Detector settings and pretrained
  checkpoint formats are preserved. On CUDA the orientation histogram now accumulates with atomics,
  so identical inputs can differ at the ulp level between calls unless
  `torch.use_deterministic_algorithms(True)` is set. (#4254)
- Replace the batched `torch.inverse` in `ellipse_to_laf` with the closed-form inverse of its lower-triangular
  2x2 matrix. Results agree with the previous implementation to ~1.6e-7 relative in `float32` (machine epsilon in
  `float64`). Reproducible numbers via `benchmarks/feature/ellipse_to_laf.py` (float32, N=1e3..1e5, torch 2.9.1,
  Linux/WSL2, base `2009933e`): throughput improves ~2.4-5.3x eager / ~3-6.5x compiled on CPU (i7-14700K) and
  ~1.4-1.7x eager / ~4.2-5.5x compiled on CUDA (RTX 4090). The largest win is MPS, where the batched inverse hit a
  pathological `linalg` path, emitted a deprecated-resize `UserWarning` on every call, and failed to `torch.compile`
  (~1500x once measured ad hoc at N=20000 on Apple Silicon; rerun the benchmark there for a citable number). The
  function now compiles with `fullgraph=True` on every backend and accepts `float16`/`bfloat16` on CPU, where the
  raw `.inverse()` call raised for low-precision dtypes (`_torch_inverse_cast` would have lifted that restriction
  too; the closed form is for speed). The off-diagonal divides `a21 = b / (a11 + a22)` by the product of the
  diagonal square roots, which can neither overflow nor round to zero: every ordering of the reciprocal product
  overflows to `inf` (or flushes a representable value to `0`) on a sufficiently lopsided or subnormal diagonal,
  and when that product is subnormal the division loses precision but never corrupts a representable result to a
  zero, `inf`, or `nan`.
- Reset an automatic augmentation policy's cached transformation matrix before applying it again, both for a
  sub-policy inside `TrivialAugment`, `RandAugment` and `AutoAugment` and for those policies nested inside
  `AugmentationSequential`. Consecutive calls that selected the same geometric policy could transform the input with
  the latest parameters while exposing the previous call's matrix through `transform_matrix`, the sub-policy
  accumulated one matrix per call for the lifetime of the module, and a nested policy never recorded its parameters
  or matrix at all. `transform_matrix` on a policy that has not run yet now returns `None` instead of raising
  `AttributeError` (#4171).
- Restore the YUV test coverage that #3539 deleted, and fix the `kornia.color` YUV docstring examples (#4045).
  `rgb_to_yuv422` and `yuv422_to_rgb` both documented an example that called the 4:2:0 function instead of
  themselves, so a reader copying either got the wrong conversion and the wrong chroma shapes; several other
  examples stated a shape in a trailing comment that disagreed with what the function returns
  (`RgbToYuv420` said `2x1x2x3` for a `(2, 2, 2, 3)` chroma plane). Every example in `kornia/color/yuv.py` now
  asserts its output shape as a doctest instead of stating it in a comment. The 4:2:2 docstrings also claimed
  the input only had to be divisible by 2 "vertical" while the guard rejects odd height *and* odd width, so a
  reader padding one axis hit a `ShapeError`, and `yuv422_to_rgb` labelled its chroma argument "UV (luma)".
  The four `ShapeError` messages those guards raise said "evenly disible by 2" and now say "divisible", as do
  the two identical messages in `kornia/color/raw.py`. `yuv422_to_rgb`/`Yuv422ToRgb` also gained a
  `.. warning::` naming a known defect: they validate only the chroma *width*, so a wrong chroma height
  surfaces as a bare `RuntimeError` from `torch.cat` (#4050). The same entry gave every YUV-to-RGB form a
  second warning, for the round trip that #4044 has since fixed; those blocks are gone, see the entry
  above. On the test side, both gradcheck skips now cover the XLA/TPU fixture as well as MPS, since XLA
  lowers a float64 request to float32, where gradcheck's default `eps=1e-6`
  makes the numerical Jacobian invalid: the name-based marker in `conftest.py` for tests called `*gradcheck*`,
  and the device guard in `BaseTester.gradcheck` for the six callers named something else. No runtime
  behavior changed.
- Stop `MultiResolutionDetector` (and therefore `KeyNetDetector`) fabricating detections, honour its `mask` argument,
  and keep its LAFs in the input dtype (#4089, #4090, #4091).

  **Padded slots are zero, and the shape is always `num_features`.** `detect_features_on_single_level` masked every
  non-candidate position to `torch.finfo(dtype).min / 2` and then ran `topk` with `k` clamped against the *pixel
  count* rather than the number of surviving candidates. Once a pyramid level ran out of above-threshold maxima,
  `topk` could no longer rank -- every remaining position carried the same sentinel -- so it returned an arbitrary
  tie-break subset, in practice the low flat indices, which is exactly the border strip `remove_borders` had just
  zeroed. `MultiResolutionDetector(BlobHessian(), num_features=100)` on a plain `64x64` image returned 70 of its 100
  features that way, each with a response of `-1.7e38` and a real-looking coordinate. Those slots are now padded
  with a zero response and a zero LAF, which is what `ScaleSpaceDetector.detect` already does for a short result,
  and `forward` re-applies that padding after the affine-shape and orientation modules, which would otherwise
  normalise a zero frame into a finite one. `detect` also only ever *trimmed* an over-long result, so a level
  capped at its own pixel count, or a per-level quota rounded down to zero, produced a short one: a one-level `8x8`
  image asking for 100 features returned 64, and the default configuration asking for 1 feature returned **0**,
  which made `lafs[0, 0]` raise. The result is now padded up to `num_features` with the same zero response and
  zero LAF. Callers that tested for the sentinel (`resp < 0`) should test `resp == 0`; callers that consumed the
  responses as scores no longer need to.

  **`num_features=1` returns a real detection.** The per-level quotas each truncate independently, and at
  `num_features=1` the six default shares are `0.508 .. 0.016`, so every one of them rounded to zero and no level
  was asked for a single candidate. The apportionment now hands its one slot to the level with the largest share
  whenever truncation would otherwise lose every slot, so `num_features=1` returns the same best feature that
  `num_features=2` returns first. An apportionment that already gives out a slot is untouched, which with the
  default configuration is every `num_features >= 2` (verified byte-identical with `torch.equal` over 16 size/count
  combinations).

  **The `mask` argument works, and means "where a detection may be".** `forward(img, mask)` and `detect(img, mask)`
  accepted a `mask`, documented it as "a mask with weights where to apply the response function", and ignored it --
  an all-zero mask returned bit-identical output. Anyone who was passing a mask and silently getting unmasked
  detections now gets masked ones. The semantics are the same in both detectors and are chosen from the caller's
  side. A boolean or integer mask is binary: any non-zero value keeps a position, so a 0/1 mask, an OpenCV 0/255
  mask and `img > 0` all mean the same thing (a raw cast would have multiplied the responses by 255 and defeated an
  absolute `score_threshold`). A floating-point mask is used as weights on the detection scores: a weight in
  `(0, 1]` scales a score toward the worst -- a non-negative score is multiplied by it, a negative one divided -- so
  a down-weighted candidate never outranks a full-weight one with an at-least-as-good score (a plain multiply pulled
  a negative score toward zero, i.e. *up* the ranking of a signed response), and a zero or negative weight
  suppresses. The weights are cast to the image dtype, so a weight a half-precision image cannot hold rounds to
  zero and suppresses, and clamped to one, so a float 0/255 mask (an OpenCV mask through `.astype(np.float32)`)
  means what the integer one means rather than scaling every score by 255. The weighting runs in float32 for
  half-precision input and is bounded at the dtype's most negative finite value, so a negative score divided by a
  small weight cannot overflow to `-inf`; an unfilled slot is ranked with `-inf` (not a finite sentinel such as
  `finfo.min / 2`, which a genuine extreme response could sort below, and did: four float32 maxima at `-2e38`
  were returned for a single image and dropped for a batch), so every finite detection precedes the padding. The
  mask is checked at the integer maximum and again at the sub-pixel-refined centre, conservatively (every octave
  pixel the centre touches), so the refinement cannot carry a detection into the zero region. A NaN weight
  suppresses rather than winning the ranking. The mask must be `(1 or B, 1, H, W)` with the
  image's spatial size. It is resampled onto every pyramid level or octave with a
  min-pool rather than an interpolation, so a zero region suppresses every level pixel it touches and a two-pixel
  zero stripe does not vanish at a factor-four level. And it is applied to the non-maxima-suppression output, not
  to the response the suppression reads: multiplying the response first carves an edge into it, and the bilinear
  ramp of that edge was a "maximum" on the suppressed side, so a blob wholly inside the zero region was still
  detected (#4102). `ScaleSpaceDetector`, which did multiply its response per octave, takes the same path; its
  detections inside the region kept by a binary mask are unchanged. A *floating-point* mask used to weight the
  response before the non-maxima suppression and the sub-pixel refinement, so a graded mask moved maxima and
  their refined positions; it now weights only the score of a maximum found in the unweighted response, so the
  set of candidates and their positions are those of the unmasked image and the weight decides their rank.

  `LocalFeatureMatcher` now carries those masks through the full extraction-and-matching path. It documented
  `mask0`/`mask1` as `(B, H, W)` inputs but never passed either to its local feature module; that historical shape is
  accepted by adding the singleton channel, and the detectors' `(B, 1, H, W)` form is also accepted directly.

  **A mask no longer changes the response dtype.** It is resampled *and cast* onto the response map, where the
  multiplication used to promote it, so a `float16` image with a `float32` mask returned `float32` responses beside
  `float16` LAFs. This also applies to `ScaleSpaceDetector`, which shares the resampling helper: a `float32` image
  with a `float64` mask used to return `float64` and now returns `float32`, i.e. the image dtype, in both detectors.
  `ScaleSpaceDetector` gains the same shape check, and accepts a boolean or integer mask through the same helper
  (it raised on one before). It resampled a mask of any size onto its octaves without complaint, so a stale mask
  from before a resize, or a transposed one, was silently stretched onto the wrong geometry, and a multi-channel
  mask was broadcast over the scale levels rather than the channels -- an error when the counts differed, the wrong
  weighting when they matched. Both detectors now require `(1 or B, 1, H, W)`.

  **`ScaleSpaceDetector` no longer leaks its own top-K sentinel for a batch.** For `B > 1` the per-octave top-K
  ranks over the whole scale-space volume with non-candidates masked to `torch.finfo(dtype).min / 2`, and an image
  with fewer maxima than requested got those sentinels back as detections: `ScaleSpaceDetector()` on
  `torch.zeros(2, 1, 32, 32)` returned 70 slots per image with a response of `-1.7e38` and an arbitrary LAF, where
  the same input at `B = 1` returned 500 zeros. Which slots a detection filled is now tracked through the top-K
  rather than inferred from the response, and an unfilled one -- the sentinel, the padding that tops a short
  result up to `num_features`, or a candidate whose frame reached outside the image -- carries a zero response and
  a zero LAF, in `detect` and, after the affine-shape and orientation modules have run, in `forward`, and sorts
  after every real detection, including a negative one. The mask is deliberately *not* `response == 0`: the
  response function is pluggable and may be signed, so an exact zero can be a genuine maximum and keeps its frame.
  The border-rejected candidate is the one visible change at `B = 1`: it used to keep its coordinates beside a
  zero response, i.e. a keypoint that was never detected, and its scale was often the largest in the result. Both
  detectors' `forward` read occupancy off the zero LAF, so a subclass overriding `detect` is honoured whole.

  **Half-precision input yields half-precision LAFs.** `detect_features_on_single_level` hardcoded `.float()` on the
  pixel coordinates, so a `float16`/`bfloat16` image came back with `float32` LAFs beside half-precision responses.
  The index-to-coordinate arithmetic runs in float32 (float64 for a float64 image) and only the finished coordinate
  is cast to the input dtype; `float32` and `float64` output is unchanged. A half-precision LAF centre carries the
  dtype's integer resolution: exact up to 256 in `bfloat16` and up to 2048 in `float16`, then on a 2 px grid up to
  twice that, 4 px up to four times, and so on -- where the old `float32` LAFs were exact.

  **A detector model must return one spatial response map.** `detect_features_on_single_level` flattened the
  whole response tensor and decoded the flat top-K index with the width alone, so a candidate from channel `c`
  landed at `y + c * H`; `BlobHessian` on an RGB image put 56 of 100 LAFs outside a `64x64` frame, up to
  `y = 179.9`. Merely decoding the channel would still spend the budget on duplicate LAFs -- a grayscale image
  repeated into RGB returned three copies of almost every feature, whose grayscale descriptors are identical.
  `MultiResolutionDetector` and `ScaleSpaceDetector` now require their detector model to emit one response channel
  and fail with a named error otherwise. This does not restrict the model input: a learned detector may consume RGB
  (or any other channel count) and collapse it to one response map itself, which is the intended color-image path.
  Single-channel input and response output are unchanged.

  **`score_threshold` must be non-negative**, and `MultiResolutionDetector.__init__` raises `ValueError` otherwise:
  non-maxima suppression writes an exact zero at every suppressed position, so a negative threshold admitted all of
  them as detections (with `score_threshold=-1.0` on a `64x64` image, 70 of 100 returned features were suppressed
  border pixels) and would now also be indistinguishable from the zero response that marks an unfilled slot.

  **`detect` enforces its documented `(1, C, H, W)`.** It is public, and a larger batch was flattened across the
  batch axis and silently returned coordinates for the wrong image. `forward` already had that guard, so only
  direct `detect` callers see a difference.

  **A short result is sorted.** `MultiResolutionDetector.detect` only ran its final top-K when the levels had
  produced *more* slots than `num_features`, so a short result came back in level order with each level's own
  padding left in place: with `pyramid_levels=2, up_levels=0, num_features=6000` on a `48x48` image the three real
  detections sat at flat indices 0, 1 and 2304. The ranking is now unconditional, as it is in
  `ScaleSpaceDetector.detect`, and the zero-response padding sorts to the end.

  **Padding is not a correspondence.** `LocalFeatureMatcher` used to describe and match every fixed-shape slot, so
  `DescriptorMatcher("nn")` returned every padded origin frame and `mnn` returned one false origin match even when
  neither image contained a detection. Zero LAFs are honest, but they are also identical, and identical descriptors
  match each other at zero distance -- with plain nearest-neighbour matching the padded block became the largest
  consistent set of "correspondences" and captured RANSAC (`SIFTFeature(400)` on two crops of one image: 0.09 px
  mean reprojection error on `main`, 31 px with the zero LAFs matched, 0.09 px with them dropped). The matcher now
  filters zero LAFs before matching, and skips a pair when either side has nothing left (`matcher` is an arbitrary
  module with no obligation to accept a `(0, D)` input); the ratio tests `match_snn` and `match_smnn` reject the
  block on their own (`match_mnn` and `match_nn` do not: identical zero descriptors are mutual nearest neighbours
  at distance zero), and
  `LocalFeature.forward` documents the one-line filter for a hand-rolled `match_nn` pipeline. The feature benchmark
  does the same before affine/orientation/description, avoiding meaningless descriptor work and the quadratic
  distance-matrix cost of padded rows; its CUDA timers also synchronize around the measured detector and full
  matching regions.
- The `Args` blocks of `ColorJitter`, `RandomBrightness` and `RandomGaussianBlur` no longer document a
  `silence_instantiation_warning` argument that none of them accepts, and `ColorJitter` now documents its
  `order` argument: a fixed (sub)set of brightness/contrast/saturation/hue indices that makes the transform
  `torch.compile` fullgraph-safe, with the drawn `_params["order"]` entry ignored when it is set.
  (#4437, #4490)
- The `Import Surface` CI check accepts deliberate public-name removals recorded in the same change.
  Edits to `tests/api_surface.json` must correspond to actual export removals and acknowledge only the exact
  recorded module/name. Submodule APIs recorded only under an ancestor, and APIs outside the inventory, use
  exact module/name entries in `tests/api_surface_removals.json`; only new entries matching a current
  `__all__` removal count. Malformed records, deleted inventory module keys, unrelated Python edits, and
  acknowledgements staged in earlier changes cannot authorize removals. Both record files trigger the workflow.
  The export resolver includes implicit submodule bindings and rejects unsupported dynamic binding expressions
  as evidence of removal. Deprecation windows and release notes still apply. (#4190, #4230)
- The generated example figures in the API reference are rendered from RGB input.
  `docs/generate_examples.py` decoded the sample images with `cv2.imdecode` and wrote them with
  `cv2.imwrite`, both BGR, so every tensor the examples fed to kornia held reversed channels. The two
  swaps cancel in the written file for channel-agnostic operations, but 35 colour-dependent figures
  change, and the ones that depend on which channel is which (`rgb_to_hsv`, `ColorJitter`,
  `apply_colormap`, the bbox and keypoint colours of `AugmentationSequential`, ...) showed a blue cast
  or swapped overlay colours. The generator now decodes and writes with PIL, draws the KeyNetAffNet
  LAF figure with kornia's own `get_laf_pts_to_draw`, and `opencv-python` and `kornia_moons` leave
  the `[docs]` extra. (#4306)
- The pinned-torch CI legs no longer run against the lockfile's `triton`. `uv sync` installs the locked torch's
  dependency set, and each matrix leg then swaps torch alone for a `+cpu` build -- whose wheels declare no `triton` at
  all -- so the lockfile's triton survived as an orphan of a torch that was no longer installed. torch guards its
  inductor triton path on import availability rather than on version, so it dispatched onto that orphan and the
  `2.6.0` leg died in `from triton.backends.compiler import AttrsDescriptor` on 32 dynamo and export tests, having
  never been green. The install step now drops sidecar packages the installed torch does not declare, and the
  verification step asserts the whole environment against that torch (`.github/scripts/check_torch_env.py`) so a
  mis-provisioned venv fails at setup instead of as a wall of unrelated-looking test failures. The five
  `tests/core/test_small_linalg.py` ONNX cases the import error was masking now export with `external_data=False`,
  which torch 2.6's dynamo exporter requires when the destination is a `BytesIO`. (#4199, #4200)
- Validate the chroma plane height in `yuv422_to_rgb`, so a chroma plane whose height does not match the luma now
  raises `ShapeError` instead of reaching `torch.cat` and dying there with a bare `RuntimeError` (#4050). 4:2:2
  subsamples width only, so chroma keeps the full luma height; the guard checked only the width ratio, where the
  4:2:0 twin `yuv420_to_rgb` checks both axes. This is observable to callers on the error path: `ShapeError` derives
  from `BaseError`, not `RuntimeError`, so code catching `RuntimeError` around `yuv422_to_rgb` to handle a malformed
  chroma plane will stop catching it. No previously working input changes behaviour — `torch.cat` already rejected
  every mismatched height. A zero-width chroma plane whose height *also* differs now reaches the guard, since the new
  clause short-circuits the width division; the matching-height case still raises
  `ZeroDivisionError` and remains open as #4056.
- `Boxes.pad`, `Boxes.unpad` and `Boxes.clamp` accept the unbatched `(N, 4, 2)` container the class
  documents and constructs. All three assumed the batched `(B, N, 4, 2)` indexing: `pad` and `unpad`
  broadcast the padding as `(B, 1, 1)`, which does not fit the unbatched `(N, 4)` coordinate view and
  raised `RuntimeError: output with shape [1, 4] doesn't match the broadcast shape [1, 1, 4]`, and `clamp`
  materialized its bounds with `repeat(1, _data.size(1), 4)` and raised `IndexError: too many indices for
  tensor of dimension 2`. Adding a leading singleton batch axis to the same boxes made all three work.
  They now broadcast against whichever rank the container holds and match the singleton-batch result;
  an unbatched container carries one image, so a per-image tensor with more than one row is rejected.
  Batched results are byte-identical, including for a non-finite bound: `clamp` stays comparison-based, so
  a `NaN` bound leaves the coordinate alone rather than propagating into it (closes #4244). (#4372)
- `Boxes.to_mask` leaves list-padding channels empty, including after coordinate transforms,
  instead of treating padding entries as real boxes. (#4390)
- `Boxes3D.from_tensor(..., validate_boxes=True)` rejects non-finite coordinates, and `validate_bbox3d`
  returns `False` for them instead of raising an `AssertionError` that names the wrong defect. This is the
  3D counterpart of #4243. An `inf` passed the positive-extent checks outright, and a `NaN` passed them
  because every comparison against `NaN` is `False`, so the box was constructed with non-finite vertices;
  in `validate_bbox3d` the `NaN` instead reached the `allclose` extent comparisons and raised
  "Boxes must have be cube, while get different widths". `validate_bbox3d`'s four internal callers use it
  for its raise and discard the result, so they now convert the `False` themselves and keep raising the same
  `AssertionError` they did before, with a message that now names the non-finite coordinates instead of
  reporting mismatched cube extents. The `validate_boxes=False` opt-out and the export gate are unchanged
  (closes #4258). (#4343)
- `Boxes3D.to_tensor` and `Boxes3D.get_boxes_shape` are differentiable again (#1396). Both reduce a box's 8
  vertices to its min/max corner with `amin`/`amax`, which is a genuine kink -- not differentiable in the
  classical sense -- wherever multiple vertices exactly tie for an axis extremum. Every face of an
  axis-aligned box has such a 4-way tie by construction, where PyTorch's `amin`/`amax` backward splits the
  gradient evenly across the tied vertices (`1/4` each): a valid subgradient, usable for optimization, but
  one that `torch.autograd.gradcheck`'s central-difference estimate does not match at that exact point --
  the same non-uniqueness any reduction has at a kink (`torch.max`, `ReLU`). That mismatch was previously
  taken for a genuine gradient bug and `to_tensor` raised `RuntimeError` whenever its input required grad,
  unconditionally, including for the common case of a caller who never differentiates through a tie at all
  (an oriented, non-axis-aligned box, or a loss that never reaches the tied component). The guard is
  removed; the docstring documents the tie behavior instead of calling it a bug. `Boxes` (2D) uses the same
  reduction and was never guarded, because an axis-aligned rectangle's ties are always 2-way, where the
  `1/2` split happens to numerically coincide with the central-difference estimate -- the same kink, just
  invisible to `gradcheck` there.
- `CameraModelBase.__init__` now validates `params` against the shape it documents, instead of storing
  whatever it is given. The typed constructors (`PinholeModel`, `BrownConradyModel`, `KannalaBrandtK3`,
  `Orthographic`) each apply the same two comparisons, so only the direct-construction path -- which is
  public, and is what the class docstring's own example uses -- was unguarded. A short vector used to
  construct and then fail with an `IndexError` from inside `AffineTransform.distort`, naming neither
  `params` nor the camera; a rank-3 `(B, 1, N)` tensor used to construct, project, and silently return a
  `(1, 1, 2)` result. Both now raise `ValueError` from the constructor. (#4316, #4369)
- `ConvQuadInterp3d` / `conv_quad_interp3d` no longer return NaN gradients for `float16` input. The
  Hessian determinant `_solve_cramer_sym3x3` divides by is a product of three second derivatives, so for
  a `[0, 1]` response it lands around 1e-4 and below and still clears the `eps` gate. The forward divides
  by it once and stays finite, but the backward scales by `1 / det**2`, which `float16` cannot represent
  (`finfo(float16).tiny` is 6.1e-5), so the gradient overflowed and reduced to NaN over part of the
  volume. The solve is now promoted to `float32` for `float16` input and the shifts cast back, as #4231
  already does for the half-precision meshgrid. `bfloat16` keeps `float32`'s exponent range and was never
  affected; `bfloat16` and `float32` outputs are unchanged. (#4305)
- `HyNet` and `SOSNet` now run in half precision, on CPU and on GPU, and no longer return NaN for a
  degenerate patch (closes #4224). Two defects sat on the same line. On CPU both raised
  `NotImplementedError: "avg_pool3d_out_frame" not implemented for 'Half'` (and the `'BFloat16'` spelling):
  their final `LocalResponseNorm` is handed a 4-D `(B, C, 1, 1)` tensor, which routes
  `torch.nn.functional.local_response_norm` through `avg_pool3d`, and that kernel has no CPU
  `float16`/`bfloat16` implementation. Separately, in `float16` on a device where the kernel does exist
  (MPS, CUDA), the descriptors came back all-NaN for any patch the network maps to exactly zero: the `eps`
  that keeps the normalisation's division defined (`SOSNet.forward`'s `eps`, `HyNet`'s `eps_l2_norm`, both
  `1e-10`) is not representable in `float16` and flushes to `0.0`, leaving `0/0`. Every `Conv2d` in `SOSNet`
  has `bias=False` and every `BatchNorm2d` is `affine=False`, so any constant patch reaches the
  normalisation as exactly zero; `HyNet` reaches it only with `is_bias=False`. `bfloat16` keeps `float32`'s
  exponent range, so the guard survives there and only the CPU kernel gap applied to it. Both models now
  take that one normalisation step in `float32` for either half-precision input and cast the result back --
  a wider lift than the `float16`-only one `kornia.feature.siftdesc` gives its own `1e-10` guards, because
  the CPU kernel gap covers both dtypes. `float32` and `float64` take the original expression and are
  bitwise unchanged on CPU, CUDA and MPS. Half-precision output does change, by 0.25-2.25 eps, and where it
  moves most it moves towards the `float64` reference rather than merely away from NaN: `SOSNet` `float16`
  goes from 2.15e-03 to 2.28e-04 of maximum absolute error against a `float64` model carrying the same
  weights, and the configurations that do not improve stay within one eps of where they were.
  Half-precision descriptors are therefore not comparable bit-for-bit across this release. (#4225)
- `ManyToManyAugmentationDispather` now rejects mismatched numbers of input bundles and
  augmentations before applying any transformation, instead of silently dropping surplus inputs
  or skipping augmentations. (#4461)
- `ModelBase.save` and `_save_outputs` write the visualizations they are given instead of raising.
  They handed `visualize`'s float output to `write_image` under a hard-coded `.png` name, but PNG is
  `uint8`/`uint16` only, so every container's `save` (`SemanticSegmentation`, `ObjectDetector`,
  `EdgeDetector`, `DepthEstimation`) raised on its first write and left an empty directory behind.
  Float images are now converted to `uint8`, clamped to `[0, 1]` first so an out-of-range
  visualization does not wrap. A batched `(B, 3, H, W)` output -- the shape the containers document
  -- is written as one file per item rather than passed whole to `write_image`, which takes
  `(3, H, W)` and rejected the rank with a message naming neither. (#4322)
- `MultiResolutionDetector.detect` apportions `num_features` across pyramid levels with largest-remainder
  (Hamilton) apportionment instead of flooring each level's fractional share independently (#4101). The floors
  used to discard every fractional part, so a small `num_features` could round the whole apportionment down: with
  the default configuration the six shares are `0.508 .. 0.016`, and every level but the finest lost its entire
  quota to truncation, so the detector effectively searched a single scale. The shortfall -- `num_features` minus
  the sum of the floors -- is now handed to the levels with the largest fractional remainder, one slot each, so
  the quotas always sum to exactly `num_features` and stay spread across scales. #4098's narrower `num_features=1`
  fallback, which handed the request's one slot to the largest-share level when every quota floored to zero, is
  the special case where the shortfall equals `num_features` and is superseded by the general apportionment.
  The apportionment was then removed again by #4222, inside this same unreleased window, so no release
  ever carries it; the `num_features=1` behaviour it fixed still holds.
- `MultiResolutionDetector.detect` now returns the `num_features` highest responses in the image (#4222). Each
  pyramid level used to be asked only for its own share of the budget, so a level could contribute at most
  `quotas[0]` (the upscaled level) or `sum(quotas[: i + 1 + up_levels])` (pyramid level `i`) -- both well under
  `num_features` for the finer levels. When an image's strongest maxima concentrated on one level, which is the
  common case rather than a corner case, the excess was never requested and so could not be ranked in by the
  final global `topk`: the detector silently returned something other than the top-k, and the sorted responses of
  `detect(k)` were not a prefix of those of `detect(k')`. Every level is now searched for `num_features` candidates, which is exactly sufficient
  because the detections a level contributes to the global top-k are that level's own strongest. The per-level
  apportionment, including #4101's largest-remainder rounding, is removed with it. This changes which keypoints
  are returned for a given image: callers get strictly better-ranked detections, and any result that depended on
  the old per-level spread will differ. The prefix is on the sorted responses, not on which detection carries a
  given response: `topk` breaks a tie by position, so exactly equal maxima can come back in a different order for
  a different `num_features`. It carries about 1.2x the candidates through the concatenation, which is not
  measurable next to the response function and the non-maxima suppression.
- `Normalize`, `Denormalize` and `Rescale` register their constants (`mean`, `std`, `factor`) as
  non-persistent buffers instead of plain attributes, so `.to(device)` moves them with the module.
  Previously they stayed on the CPU: eager tolerates the mix, but `torch.export` traces with fake
  tensors and refused it, so exporting a preprocessing pipeline from an accelerator failed while
  the same pipeline exported fine from the CPU. `Denormalize` now coerces a scalar `mean`/`std` to
  a 1-D tensor, as `Normalize` already did, which changes its `__repr__` to match `Normalize`'s and
  lets a scalar `Denormalize` reach the ONNX export branch instead of raising `IndexError` there.
  The buffers are non-persistent, so `state_dict()` is unchanged and existing checkpoints still
  load. (#4323, #4330)
- `PatchSequential` now augments every patch of every image and runs the complete selected sequence.
  The padding, reconstruction and parameter-generation behaviour changes are listed under
  *Breaking changes*. Refs #4421. (#4460)
- `PinholeCamera.from_parameters` now fills `height` and `width` for the whole batch. It built them with
  `height_tmp[..., 0] += height` on a `(B,)` zero tensor, so every batch element after the first kept `0`
  while `fx`, `fy`, `cx`, `cy`, `tx`, `ty` and `tz` were broadcast correctly, and the camera looked healthy
  until something read its image size. `batch_size=1`, the only case that worked, is unchanged. (#4279)
- `PinholeCamera.project` now reports rank-1 inputs with the same explicit `ValueError` used by the point-conversion
  helpers instead of an internal `IndexError`. Refs #4266. (#4450)
- `PinholeCamera.scale` no longer shares its `extrinsics` tensor with the source camera, so
  `scaled.tx = 7` leaves the camera `scale()` was called on unchanged. The returned camera's numbers are unchanged. (#4264, #4349)
- `PinholeCamera`, `StereoCamera`, and `warp_frame_depth` now support empty batches. (#4386)
- `RandAugment`'s `m` guard is exclusive at both ends, but its docstring and its error message
  both named the closed interval `[0, 30]`, so a user who asked for the maximum strength the
  message advertised got an exception saying `30` was in range. Both now read `(0, 30)`; the
  accepted values are unchanged. `n` was validated nowhere: `n=0` constructed a `RandAugment`
  that applied nothing, and `n` above the policy length was silently clamped, because the
  sampler draws without replacement. It is now checked against the policy. `AutoAugment`'s
  magnitude bin indexes two adjacent points of an 11-point scale, so `9` is the last usable
  bin; a larger one raised a raw `IndexError` naming an internal tensor, and a negative one
  wrapped silently onto a reversed range. Out-of-range bins now name the operation and the
  valid range. Operations that ignore the magnitude entirely keep accepting any bin. (#4447)
- `RandAugment`, `AutoAugment`, `TrivialAugment` and `AugMix` no longer silently upcast half-precision
  batches to `float32`. `OperationBase.forward` — the shared gate every auto-augment op routes through —
  moved its `batch_prob` mask to `input.device` but not `input.dtype`, and since `batch_prob` is `float32`
  the expression `batch_prob * op(x) + (1 - batch_prob) * x` promoted a `float16` / `bfloat16` input up to
  `float32` (the op output and input are both low precision, only the gate was not). `float32` / `float64`
  inputs are unaffected and their numeric output is byte-identical. The mask is now cast to `input.dtype`
  as well. Partially addresses #4153: this clears the `RuntimeError` behind nineteen of that issue's Linux
  CPU half-precision xfail lines for `tests/augmentation/test_auto_operation.py` (all ten `cpu_bfloat16`
  entries, nine of the ten `cpu_float16`), which are removed from `testing/half_precision_xfails/`. The
  tenth `float16` line is kept: `TestAutoAugment::test_reproduce[cpu-float16]` still fails, now on an
  `AssertionError` rather than a `RuntimeError` — a smaller residual, not a fix. (#4210)
- `RandomBoxBlur`'s `normalized` argument is documented as what it does. It was described as "if True, L1 norm
  of the kernel is set to 1", but it was forwarded positionally into `kornia.filters.box_blur`'s `separable`
  parameter: it chooses between the separable and the single 2D pass, and the kernel is L1-normalized either
  way, so `normalized=False` returns window means, not sums. The call now passes `border_type` and
  `separable` by keyword; outputs are unchanged. (#4433, #4486)
- `RandomChannelDropout.fill_value` and `RandomGaussianBlurGenerator.sigma` (when passed as a
  Tensor) are registered as non-persistent buffers instead of plain attributes, so they move with
  `Module.to()`/`.half()`/`.cuda()` and appear in `named_buffers()` like the rest of the module's
  state; `state_dict()` keys are unchanged. Neither crashed before this change (both re-derived
  device and dtype inline on every call), so this is a hygiene fix, not a bug fix for a crash.
  One user-visible effect: after `.half()`/`.bfloat16()` the fill value is now held in that dtype,
  so an input already in the module's dtype is unaffected, while a `float32` input fed to a
  half-converted `RandomChannelDropout(fill_value=0.3)` is filled with `0.300048828125` (float16)
  or `0.30078125` (bfloat16) rather than `0.3` -- the same buffer semantics as `Normalize`.
  (#4337)
- `RandomCutMixV2` and `CutmixGenerator` document `cut_size` as what it is: the `[min, max]` clamp on the
  Beta-sampled mixing coefficient `lambda`, where the cut side is `floor(sqrt(1 - lambda) * side)`, so a larger
  `cut_size` gives a smaller cut. It was described as the "minimum and maximum cut ratio". A minimum of `1.0` is
  now rejected with a `ValueError`: it forced `lambda = 1`, built an inverted zero-size box and silently made the
  augmentation an identity. (#4439, #4491)
- `RandomMosaic` now preserves `(H, W)` for non-square inputs when `output_size=None`; it previously
  returned `(W, H)`. `start_ratio_range` now scales x by width and y by height; it previously scaled
  x by height and y by width. (#4459)
- `RandomRain` now rejects drop heights equal to the image height and absolute drop widths equal to the image
  width with the documented validation error, instead of allowing boundary-sized drops to reach an internal
  `IndexError`. (#4451)
- `RandomRain` with `same_on_batch=True` now samples the same number of rain drops for all samples in the batch. (#4453)
- `RandomThinPlateSpline(scale=0.0)` now generates zero control-point displacement instead of
  raising an error when constructing a degenerate uniform distribution. (#4463)
- `RandomTransplantation` and `RandomTransplantation3D` transplanted nothing on MPS when no `excluded_labels`
  were given: PyTorch's MPS backend evaluates `all()` over the empty excluded-label axis to an undefined value,
  usually `False`, so every donor label was filtered out and the output equalled the input. The filter is now skipped when there is
  nothing to exclude. (#4160)
- `RenderingDeFMO` (used by `DeFMO`) no longer crashes on a half-precision forward pass. Its rendering
  time-steps (`times`) were a plain Python attribute, not a registered buffer, so `nn.Module.to()` never
  moved it; `forward` re-derived `times`'s device from the input on every call but never its dtype, so
  `RenderingDeFMO().half()` left `times` at `float32` and the first forward raised `RuntimeError: Input
  type (torch.FloatTensor) and weight type (torch.HalfTensor) should be the same`. `times` is now a
  non-persistent buffer (it is fully determined by `tsr_steps`, not learned, so `state_dict()` keys and
  existing checkpoints are unaffected), and `forward` casts into a local variable rather than depending
  on the caller having matched dtype. Same bug shape as #4069/#4079 in a different class; the normal
  `float32` path is byte-identical to before. (#4319)
- `SIFTDescriptor`, `DenseSIFTDescriptor`, `HardNet` and `HardNet8` return NaN for a constant patch in `float16`,
  and the SIFT descriptors' backward is NaN in `float16` wherever two neighbouring pixels are equal.

  `F.normalize`'s default `eps` of 1e-12 is not representable in `float16` and rounds to zero, so the
  `norm.clamp_min(eps)` that exists to stop a zero-norm input becoming `0 / 0` was itself zero in exactly that
  dtype: an all-zero patch normalised to NaN, while `bfloat16`, `float32` and `float64` were fine. A detector that
  pads a short result hands the descriptor a zero LAF, which samples one image point repeatedly and produces
  exactly that patch -- `ScaleSpaceDetector(400)` on a random `64x64` `float16` image gave 365 NaN descriptor rows
  -- and a NaN descriptor is worse than a meaningless one, because it propagates through `torch.cdist` and poisons
  the whole matching. The nine `F.normalize` calls in those four modules now go through one helper that
  normalises a `float16` input in float32 and casts back (an `eps` representable in `float16` is safe but not
  neutral: a norm in the subnormal window came back as 0.5). `bfloat16`, `float32` and `float64` results are
  unchanged: they keep the 1e-12 default.

  The forward pass was only half of it. `SIFTDescriptor` and `DenseSIFTDescriptor` guard `sqrt(gx^2 + gy^2 + eps)`
  and `atan2(gy, gx + eps)` with `eps = 1e-10`, which is zero in `float16` -- and a squared `float16` gradient
  underflows long before that -- so at every pixel with a zero gradient both sat on their singular point and the
  input gradient came back NaN (9 of 4096 on ordinary random `32x32` patches). The gradient magnitude and
  orientation are now computed in float32 for `float16` input and cast back, and the RootSIFT `sqrt` is computed
  in float32 for `float16` as well. An *exactly* flat patch -- what a zero-LAF padding slot samples -- was the
  remaining case: the guard gave every zero-gradient pixel a magnitude of `sqrt(eps)`, so a flat patch's
  descriptor was a unit vector built from `eps` (float32) or a subnormal one (float16) whose `1 / norm` gradient,
  together with `atan2`'s `1 / eps`, overflowed through the float16 cast into an all-NaN input gradient (and was
  a meaningless ~1e8 in float32). A zero-gradient pixel now contributes a zero magnitude, a zero vector normalises
  to zero with a zero gradient (the `eps` clamp's `1 / eps` was never meaningful), and
  `PatchDominantGradientOrientation` skips its parabolic refinement on an empty or uniform histogram instead of
  dividing `0 / 0`; a flat patch has a zero SIFT descriptor and a zero input gradient in every dtype. `bfloat16`,
  whose exponent range holds both the guard and the squares, and the wider dtypes are byte-identical at every
  pixel whose gradient is not exactly zero; a patch with such pixels (a saturated region) loses their `sqrt(eps)`
  contribution, a change of at most `1e-5` per pixel before normalisation.
- `SemanticSegmentation.visualize` works for CUDA, MPS and half-precision models. It indexed the
  CPU-drawn colormap with the mask's `argmax`, which raises for CUDA and MPS masks, and recognised a
  softmax head with `torch.allclose(sum, 1)` at float32-sized default tolerances, which a float16 or
  bfloat16 sum of probabilities never meets; the tolerance now scales with the number of classes and the
  dtype's epsilon, and the colormap follows the mask's device and dtype, so the visualization keeps the
  model's dtype instead of coming back as float32. `OnnxLightGlue` without `onnxruntime` raises an
  `ImportError` that names `pip install "kornia[onnx]"`, like the lazy-loader handles do, instead of a
  bare `BaseError`. (#4301)
- `So3.exp`, `So3.log`, `Se3.exp`, `Se3.log` and `Se2.exp` return finite gradients at the identity,
  and `So3.log` also at a half turn, instead of `nan` on every dtype. The forward values are unchanged;
  `torch.where` was differentiating the singular branch it does not select. (#4405)
- `StereoCamera` now rejects projection matrices whose last two dimensions are not `(3, 4)` with a
  `StereoException` naming the invalid camera. Previously, the shape guards never fired, so malformed
  matrices could be accepted or fail later with an unrelated tensor error. (#4385)
- `VideoBoxes.get_boxes_shape()` and `VideoBoxes.to_mask()` no longer raise `TypeError`. Both are
  inherited from `Boxes` and pass `as_padded_sequence=True` to `to_tensor`, but the `VideoBoxes`
  override declared only `mode`, so every call on a `VideoBoxes` failed -- including the containers
  `AugmentationSequential` builds for video box inputs. The override now accepts and forwards the
  keyword; it only changes a list-backed container, and a `VideoBoxes` built from a
  `(B, T, N, 4, 2)` tensor is not one, so no working call changes its result.
  (#4176, #4365)
- `VisualPrompter.predict()` called without any prompt (the "run the prediction without prompts" example on the
  Segment Anything page) raised `TypeError: object of type 'NoneType' has no len()` from the prompt augmentation
  container; it now predicts from the image embedding alone, as documented. (#4178)
- `Z1Projection.unproject` now materialises a python `int`/`float` `depth` on the device and
  in the dtype of `points`; it used to build a CPU float32 tensor, which raised `RuntimeError`
  on every accelerator and widened float16/bfloat16 results to float32. (#4340)
- `_cdist` replaces `dm.clamp(min=0.0).sqrt()` with a masked argument substitution, avoiding `NaN` gradients for
  exact zero-distance pairs on PyTorch 2.5.1 and 2.9.1 where `clamp`'s boundary gradient passed through to `sqrt(0)`.
  PyTorch 2.14.0 was already finite, and forward behavior and computed distances are unchanged. (#4233)
- `add_weighted` now keeps Python scalar weights in operator math precision instead of rounding them to the input
  dtype before arithmetic. This improves `float16` and `bfloat16` accuracy and makes fractional scalar weights on
  integer inputs produce the correctly promoted floating-point result instead of being truncated. (#4154)
- `axis_angle_to_rotation_matrix` accepts the `(*, 3)` shape its own guard message promises, instead of
  only `(N, 3)`. The body did `wxyz.unbind(dim=1)` and `.view(-1, 3, 3)`, so an unbatched `(3,)` raised
  `IndexError: Dimension out of range` and any extra batch dimension raised `ValueError` out of the unbind,
  three frames below kornia's own shape guard. Every sibling conversion in the module already accepted
  `(*, 3)`, so `axis_angle_to_rotation_matrix(rotation_matrix_to_axis_angle(R))` composed for an `(N, 3, 3)`
  rotation matrix and for nothing else, including for the shape `rotation_matrix_to_axis_angle`'s own
  doctest returns. This is additive: `(N, 3)` output and gradients are byte-identical, and only shapes that
  used to raise now return (closes #3955). (#4342)
- `bbox_to_mask3d` now preserves the input dtype in the returned mask, matching `bbox_to_mask`
  and the `Boxes3D.to_mask` contract: the earlier implementation downcast every result to
  `float32`, so a `float64` or half-precision box produced a mask that silently lost precision.
  The mask keeps the input box's dtype end to end. (#4376)
- `bbox_to_mask3d` now returns the intersection of the three axis ranges for a box that covers or
  overhangs a whole output axis, matching `Boxes3D.to_mask`, instead of filling the entire volume
  (#4255, #4303). The old implementation `|`-ed the three broadcast axis slabs and tried to recover the
  intersection with a three-way `all()` reduction; that recovery breaks the moment any one slab
  covers a whole axis, since the union is then all-true on that axis and the reduction can no
  longer see the other two slabs' bounds. Computing the intersection directly with `&` needs no
  such recovery step. Interior boxes (the case the old reductions happened to recover correctly)
  are unaffected and byte-identical.
- `bbox_to_mask` built its pixel-position grid in the box dtype. With `float16` boxes (and
  `bfloat16` boxes) on images wider or taller than 2048 px (256 px for `bfloat16`) consecutive
  pixel positions collapsed, so rows and columns near the collapse were mismasked; the grid is
  now built in `float32` and `float16`/`bfloat16` results are byte-identical to `float32`/`float64`.
  `RandomErasing` and `RandomCutMixV2` build their masks through it. (#4336)
- `depth_from_disparity` accepts Python integers and scalar tensors for the baseline and focal length. (#4392)
- `depth_from_plane_equation` returns a finite depth for a ray exactly parallel to the plane. The
  near-singular guard was `eps * torch.sign(denom)`, and `torch.sign` is zero at zero, so at the exact
  singularity the epsilon was multiplied away and the division still ran against zero, returning `inf`.
  A grazing ray is not an exotic input: it is every pixel on the horizon of a ground plane. The guard
  now picks its sign with a comparison, which has no such hole and keeps the sign the small non-zero
  denominators already got. `torch.copysign` reads the same but is not exportable, and this function
  is in the documented ONNX export surface, so the comparison form is pinned in
  `tests/onnx/test_export_coverage.py`. The finite-depth promise is bounded by the dtype: the default
  `eps=1e-8` is below float16 resolution, so half-precision callers must pass a representable `eps`.
  (#4280, #4348)
- `depth_to_normals` now raises `ShapeError` when `H < 2` or `W < 2`, since surface normals need two
  tangent directions. Previously, singleton axes could yield zero or non-finite normals, and empty
  spatial dimensions failed inside padding. Inputs with both dimensions at least 2 are unchanged. (#4458)
- `distance_transform` no longer returns NaN gradients when the cascade's convolution is exactly zero, including
  sparse masks and all-zero inputs; the existing forward output is unchanged. (#4232)
- `distance_transform` no longer silently returns a wrong or all-zero result when its `exp(-dist/h)` kernel
  underflows (#4152). The kernel's axis-aligned tap, at distance `kernel_size // 2` along a single axis,
  underflowing to exactly zero -- reachable from the documented default `h=0.35` at `kernel_size >= 15` in
  `float16` -- used to make the convolution silently drop it, returning a plausible-looking but wrong distance
  field or, with a smaller `h`, an all-zero one. Two changes: `float16`/`bfloat16` inputs now run the cascade in
  `float32` and cast the result back (the default `h=0.35` case is fixed by this alone, matching `float32`'s own
  working range); and an `h`/`kernel_size`/dtype combination whose axis-aligned tap still underflows
  `torch.finfo(dtype).tiny` now raises instead of guessing. This also catches a case that is not itself
  numerically wrong on CPU (`h=0.01` at `kernel_size=3` in `float32` tracks the analytic reference to ~2e-4,
  carried through the subnormal range) but where MPS flushes subnormals to zero while CPU does not, so the same
  call used to silently disagree across backends; raising there rules out that disagreement rather than trusting
  which backend a caller happens to run on. Concretely, at the documented default `h=0.35` in `float32`,
  `kernel_size >= 63` now raises: CPU used to return a usable-looking answer there while MPS silently returned
  something else for the same call.
- `download_file_from_url` and `download_hf_file` take an optional `validate=` callable, and
  `kornia.core.check_safetensors` is the one the KimiVL and SigLIP2 builders pass. A transfer cut
  short after a 2xx status leaves a truncated file in the cache, which was then returned as a cache
  hit on every later call -- `from_pretrained_hf()` failed until the user deleted it by hand.
  `validate` gives a download-only call the quarantine `load_state_dict_from_url` gets from its load
  step: a rejected entry is moved aside, the next source is tried, and the discarded source is
  re-fetched once. Without `validate` the behaviour is unchanged. (#4309, #4332)
- `elastic_transform2d` now builds its identity sampling grid with the requested `align_corners`
  convention, so a zero displacement field preserves the input image. (closes #4235). (#4382)
- `extract_patches_simple` and `extract_patches_from_pyramid` return an all-zero patch and a zero LAF gradient for a
  non-finite LAF frame instead of handing `grid_sample` an invalid grid. A frame holding a NaN or an infinity
  anywhere -- including only in its center -- is detected and sanitized before any grid arithmetic, and finite frames
  in the same batch are untouched. Such a frame previously produced a finite-looking border-sampled patch, and its
  backward pass could terminate the process inside torch's CPU `grid_sampler_2d_backward` kernel with
  `padding_mode="border"`; a training-time detector that emits a degenerate LAF hits exactly that path through
  `LAFOrienter`, `LAFAffNetShapeEstimator` and `LAFDescriptor`.
- `fft_conv` now accepts CPU float16 and bfloat16 inputs by computing the FFTs
  in float32 and returning the input dtype. (#4394)
- `find_homography_dlt(..., solver="lu")` now solves the minimal four-point system with an adaptive
  homogeneous gauge instead of applying LU to its singular normal matrix. This is the estimator
  `RANSAC(model_type="homography")` calls for every minimal sample, so homography RANSAC results
  change on every platform, not only where the old formulation broke down: over 2048 minimal
  samples the median transfer residual drops from 2.3e-5 to 1.9e-6 and the worst case from 4.2e+5
  to 4.5e+2. It also fixes the all-NaN homographies seen for a four-point sample on Windows with
  PyTorch 2.14, and removes the corresponding CPU half-precision known failures. A four-point
  sample containing a zero weight now returns a finite homography instead of NaN. Samples of five
  or more points keep the previous normal-equation formulation and are numerically unchanged.
  (#4197)
- `import kornia` no longer imports onnxruntime when it is installed (#4295)
- `infer_bbox_shape3d` and `bbox_to_mask3d` reject rank-4 `(B, N, 8, 3)` input with a `ShapeError`,
  the way `infer_bbox_shape` and `bbox_to_mask` have since #4218. `validate_bbox3d` accepts the rank-4
  form and reshapes internally, but both callers index dim 1 as the vertex axis, so the box axis was
  read as the vertices: an out-of-bounds error with one box, and three `(1, 3)` tensors -- one value
  per coordinate rather than per box -- with eight. (#4248, #4351)
- `infer_bbox_shape` and `bbox_to_mask` now reject rank-4 `(B, N, 4, 2)` inputs with `ShapeError`. Previously,
  `N < 3` raised an incidental `IndexError`, while `N >= 3` silently returned `(B, 2)` extents computed from the
  wrong vertices. (#4218)
- `iterative_quad_interp3d`'s `max_candidates` cap is now a per-image budget rather than one shared
  across the batch. The `topk` ranked the flattened `(B*C)` candidate list, so an image's refined
  keypoints depended on which other images shared its batch: a quiet image next to a high-contrast
  one got none. This also reaches `IterativeQuadInterp3d` and `AdaptiveQuadInterp3d` in `patch` mode.
  The docstring already read as a per-image budget. A negative `max_candidates` now raises
  `ValueError`; it previously raised a `topk` `RuntimeError`, and under the per-image ranking it
  would otherwise have silently disabled refinement instead. (#4256, #4350)
- `jpeg_codec_differentiable` no longer returns an all-`NaN` image at `jpeg_quality=0`, a value inside the
  `[0, 100]` range it documents and validates. `_jpeg_quality_to_scale` divides `5000` by the quality on the
  `< 50` branch, which is `inf` at `0`, and the polynomial floor of `inf` is `NaN`, poisoning the image and
  its gradient. A quality of exactly `0` is now given the scale of quality `1`, which is the table libjpeg
  gives quality `0`. The guard is that one point: a fractional quality in `(0, 1)` was already finite, keeps
  its own larger scale, and is untouched, so this endpoint is not the limit of the formula from above.
  Output and gradients for every quality `> 0` are byte-identical. The tests drew
  the quality with `torch.randint(low=0, high=100)` unseeded at sixteen sites, so roughly one run in twenty of
  `tests/enhance/test_jpeg.py` went red on any job; the draws now start at `1` and the boundary has pinned
  cases of its own. (#4205)
- `kornia.contrib.diamond_square` generates fractals with a spatial side below 3 px. A 1 px side raised
  `ValueError: math domain error` and a 2 px side a `TypeError` from a float in the `torch.rand` size, so
  `RandomPlasmaBrightness`, `RandomPlasmaContrast` and `RandomPlasmaShadow` failed on such images. A small
  side is now drawn on the 3 px grid and sliced; outputs for sides of 3 px and more are bit-identical.
  (#4472, #4488)
- `kornia.contrib.super_resolution` builders construct again. `SuperResolution` never implemented
  `ModelBase`'s abstract `from_config` and defined no `__init__`, so `SmallSRBuilder.build()` and
  `RRDBNetBuilder.build()` both raised `TypeError` at construction — the whole public
  super-resolution entry point had been unreachable since the models refactor made `from_config`
  abstract. It now has a `SuperResolutionConfig` and a `from_config` that dispatches to either
  builder family, and both builders are covered by tests. Fixes #4291. (#4335)
- `kornia.enhance.equalize`, `equalize3d`, `RandomEqualize` and `RandomEqualize3D` raise a `RuntimeError` naming
  the `[0, 1]` input range for values the 256-bin lookup cannot index, instead of a raw
  `index 259 is out of bounds for dimension 1 with size 256` from the gather. The check uses
  `torch._assert_async`, so it adds no device sync and `torch.compile` fullgraph still works, and inputs that
  equalized before (including values a hair above 1) are unchanged. The docstrings now state the range, the
  256-bin histogram, and that a 3D volume of at most 255 voxels per channel comes back unchanged.
  (#4431, #4432, #4489)
- `kornia.feature.LightGlue` no longer probes for `flash_attn`; the SDPA path it always used is now
  the only one. `kornia.feature.lightglue.Attention.enable_flash`, the attribute that combined the
  constructor argument with the probe result (and so always equalled the argument, because SDPA is
  present on every supported torch), was renamed to `Attention.allow_flash`. (#4287)
- `kornia.io.get_sample_images` and `kornia.onnx.ONNXLoader.list_operators` / `list_models` no longer need
  `requests`, which `pip install kornia` never installed; they fetch with the standard library `urllib` instead.
  The failure exceptions change with it, all still `OSError` subclasses: a sample-image URL that returns 404 now
  raises `urllib.error.HTTPError` instead of `PIL.UnidentifiedImageError`, and an unreachable host raises
  `urllib.error.URLError` instead of `requests.exceptions.ConnectionError`. A 404 from the Hugging Face listing
  still raises the same `ValueError` (#4302)
- `kornia.io.load_image` and `write_image` work on the kornia_rs that a plain `pip install kornia`
  resolves. kornia_rs 0.1.11 moved its image readers and writers from the package root into
  `kornia_rs.io`, and kornia kept calling the root, so on 0.1.11 and newer `load_image` raised
  `AttributeError` for every JPEG and every PNG that is not plain RGB, and `write_image` raised for
  TIFF, 16-bit and float32 output (8-bit JPEG and PNG writes still worked); the lock's 0.1.10 kept CI
  green. kornia now calls `kornia_rs.io` and requires a kornia_rs that provides it (see *Breaking changes*). That
  decoder also reads 16-bit PNG/TIFF and float TIFF, which 0.1.10 rejected: `ImageLoadType.UNCHANGED`
  returns them as `uint16`/`float32`, and the 8-bit load types raise a `NotImplementedError` that says
  so instead of mislabelling the tensor. The nightly PyPI job now imports the installed wheel from
  outside the checkout (it used to import the source tree) and round-trips an image instead of only
  importing. (#4326)
- `nms2d` accepts a non-square `kernel_size`. It used to raise `RuntimeError: shape '[1, 1, -1, H, W]' is
  invalid for input of size ...`, because the neighbourhood kernel was built with its two extents swapped
  and the padding was applied to the wrong pair of edges. (#4240, #4242)
- `nms3d` accepts any `kernel_size`. Every size other than `(3, 3, 3)`, the one served by a hand-written
  branch, used to raise: `_compute_zero_padding3d` defined a `(k - 1) // 2` helper and then returned the
  full kernel sizes, so the padded volume did not match the kernel it was convolved with. (#4241, #4242)
- `normalize_laf`, `denormalize_laf` and `generate_patch_grid_from_normalized_LAF` count a singleton image axis as
  one pixel of extent instead of dividing by `size - 1 == 0`. `normalize_laf` raised `ZeroDivisionError` for a
  1-pixel-wide or 1-pixel-tall image -- and with it both patch extractors on their default
  `normalize_lafs_before_extraction=True` path -- while `denormalize_laf` silently collapsed every LAF to zero.
  The conversions are now finite and round-trip, and both extractors return finite patches for such an image.
- `normalize_quaternion` floors a positive `eps` at the smallest positive `float16` subnormal, so its default
  `eps=1e-12` remains an active division guard and a zero quaternion returns finite zeros instead of NaNs without
  changing non-zero representable `float16` magnitudes. An explicit `eps=0.0` still disables the guard. This also
  keeps `Quaternion.normalize`, `quaternion_to_rotation_matrix`, and their downstream pose conversions finite for
  that input, matching the existing `float32`, `float64`, and `bfloat16` behavior (#4162).
- `packaging` is no longer a runtime dependency; it was never imported. The `dev` extra no longer lists
  `pytest-cov` (CI runs `coverage run -m pytest`), `ruff` (the pre-commit hook installs the pinned copy) or a
  `numpy<3` cap, and `uv.lock` is regenerated to match (#4296).
- `pixel2cam` now rejects depth tensors outside the documented `Bx1xHxW` shape. The guard's predicate
  bound as `(ndim != 4) and (shape[1] == 1)`, so four-dimensional multi-channel depth bypassed the
  channel check — a three-channel depth silently broadcast into separate camera coordinates — while
  scalar and one-dimensional inputs raised `IndexError` from the guard itself. Code that passed those
  shapes and relied on them being accepted now raises `ValueError`. Valid `Bx1xHxW` depth is unaffected
  and its output is unchanged. (#4314)
- `pixel2cam` now validates the full `Bx4x4` shape of `intrinsics_inv`, rejecting invalid matrix sizes
  before they cause unrelated transformation errors or return the wrong number of coordinate components. (#4381)
- `quaternion_exp_to_log` and `euler_from_quaternion` no longer return `nan`/`inf` gradients at their
  respective rotation singularities (#4007). Both differentiate an inverse trig function (`acos`, `asin`)
  whose own derivative is unbounded at its domain boundary (`w = +-1`, `sinp = +-1`); `quaternion_exp_to_log`
  hits that boundary exactly at the identity quaternion `(1, 0, 0, 0)` -- the standard initialization for
  pose optimization -- where the unbounded derivative used to multiply the identity's exactly-zero vector
  part into `0 * inf = nan`, and `euler_from_quaternion`'s `pitch` hits it at gimbal lock. Both now guard the
  argument fed to the differentiated `acos`/`asin` call away from the boundary while taking the returned
  *value* from a detached copy of the real (possibly boundary) argument, so the forward result is unchanged
  -- verified byte-identical against the previous implementation over 500 random inputs including several
  forced onto the exact boundary -- while the gradient no longer depends on what a given torch version or
  backend's own `acos`/`asin` derivative happens to return there. `quaternion_exp_to_log`'s antipode
  `(-1, 0, 0, 0)` still returns a large but finite gradient (`~pi/eps`) from an unrelated, pre-existing
  division by its clamped (zero) norm; that is unchanged and out of scope here.
- `quaternion_to_axis_angle` returns the correct gradient at every identity-like quaternion, not only the
  positive unit identity `(1, 0, 0, 0)` (#4237). Its zero-vector-part branch used the constant `k = 2.0`,
  which is only the `w = 1` case of the true analytic limit `k = 2 / w` (`w` = the real component); every
  other identity-like input got a wrong gradient. That included the negative unit identity `(-1, 0, 0, 0)`
  -- the same physical rotation as `(1, 0, 0, 0)` under the double cover -- which got the **wrong sign**
  (`+2` instead of `-2`), and any non-unit "identity" such as `(2, 0, 0, 0)` (a scale this function
  explicitly permits), which got the **wrong magnitude** (`+2` instead of `+1`). This was a genuine
  backward-correctness defect, not a missing edge case: a valid unit quaternion could produce the opposite
  gradient direction in pose optimization depending purely on which sign of the identity it started from.
  The prior regression test for the `#3949` NaN-gradient fix only exercised `w = 1`, the one point where
  the wrong constant and the correct formula happen to agree, and so could not have caught this. The
  division needed for `2 / w` is guarded at `w = 0` the same way the function's existing `sin_theta`
  division already is. That gate is `~pos` -- the mask that actually selects the branch -- and not merely
  `w != 0`: `2.0 / t` lowers to `t.reciprocal() * 2`, whose backward is `-grad * result**2`, and for a
  rotation just short of a half turn `w` is small but non-zero, so `(1 / w)**2` overflows to `inf` and
  meets the exact `0.0` that the unselected branch receives as `0 * inf` -> `nan`. In `float16` that
  would have been every rotation within ~0.45 degrees of 180 -- ordinary inputs, not degenerate ones.
  The `2 / w` coefficient is also detached, because on the branch that selects it the vector part is
  zero, so `d(out)/dw` is exactly zero there and computing it through `-2 / w**2` only reintroduces the
  same overflow.

  The fully degenerate all-zero quaternion `(0, 0, 0, 0)` -- not a valid rotation -- keeps its
  `(0, 0, 0)` forward value. Its gradient there changes, and improves: it was `(0, 2, 2, 2)` on torch
  2.14 but already `(nan, 2, 2, 2)` on torch <= 2.9.1, where `atan2(0, 0)`'s derivative with respect to
  its second argument -- a `0 / 0` -- returns `nan`. `atan2` is now shielded across that whole masked
  branch, which also clears a pre-existing `nan` for a zero vector part whose `w**2` underflows
  (`w = 1e-30` in `float32`, `|w| < 2.4e-4` in `float16`). No gradient is analytically correct at a point
  with no well-defined limit, so the changed value is not a regression.

  The forward value is unaffected everywhere: the `atan2` shield deliberately takes its value from the
  unshielded expression, because routing `cos_theta` through `torch.where` hands `atan2` a contiguous
  tensor instead of a stride-4 view and can select a different kernel. Verified byte-identical against
  the previous implementation over 2000 random inputs in `float64`, `float32`, `float16` and `bfloat16`,
  including forced unit, negative-unit, scaled, all-zero and near-half-turn quaternions, with no
  non-finite gradient row remaining at any of those dtypes.
- `rad2deg` and `deg2rad` now handle integer tensor inputs correctly and preserve
  float64 precision. `angle_to_rotation_matrix` inherits the corrected conversion,
  while the implementation preserves ONNX export compatibility. (#4358)
- `reproject_disparity_to_3D` (the `StereoCamera` method and the module-level function, which share
  one body) no longer transposes the two pixel indices. The pixel meshgrid was unbound as
  `v, u = torch.unbind(uv, dim=-1)`, but `create_meshgrid(normalized_coordinates=False)` returns
  `(x, y)`, so the column fed `v` and the row fed `u`: `X` was computed from the row and `Y` from
  the column. Every returned point was the value belonging to its transposed pixel -- kornia's
  answer at `(row, col)` was what `cv2.reprojectImageTo3D` puts at `(col, row)`, the OpenCV
  semantics this function was added to provide (#2042). A square rig with `fx == fy` and
  `cx == cy` agrees only on the diagonal, so the error was silent rather than absent. Output
  changes at every pixel with `row != col`, square inputs included. The real-data regression
  fixture stored ten points as ten rows of one column while its own comment and its ground truth
  describe one row of ten columns,
  which is why it passed against the swapped code; it is now laid out as the comment says, and
  the fixed code reproduces the unchanged ground-truth values. #4317's two wart pins for this defect
  are retired and its strict-xfail convention test is now an ordinary passing regression. (#4269, #4366)
- `resize` and `rescale` return an empty image for a zero-sized output instead of raising
  `ZeroDivisionError` out of the scale-factor computation (#4115). `resize(x, (0, 4))`, `resize(x, 0)`,
  `Resize((0, 4))`, and any `rescale` factor that takes a side to zero (`rescale(x, 0.0)`) previously died
  in `h / size[0]`; they now return an empty tensor of the requested shape, which is the answer
  `warp_affine`, `warp_perspective` and `center_crop` already gave for the same request. The empty result
  stays connected to the autograd graph, the way `_empty_warp_output_2d` builds it for the warping ops, so
  a batch element that degenerates to a zero-sized output still contributes a zero gradient rather than
  detaching.

  A degenerate *input* is a different question and is now rejected rather than divided by: resizing a
  `(1, 3, 0, 4)` image to a positive size, or with an `int` size at all -- an `int` names one side and takes
  the other from an aspect ratio a degenerate input does not have -- raises
  `ValueError("Input image size must be positive. Got height=0, width=4.")`, the message the warping ops
  already use, instead of the same bare `ZeroDivisionError`. Empty in and empty out is well defined and
  still comes back empty. (#4140)
- `solve_cubic` and `solve_quartic` return finite gradients for a row whose neighbours in the
  same batch take a different branch. The `D > 0` branch selected its rows with `abs(R) > 1e-16`
  alone, so it evaluated `sqrt(D)` on `D < 0` rows too. Those rows are never read back, but the
  resulting `-Q / nan` stays in the graph and its backward returns `nan`, which reaches every
  coefficient: a three-real-root cubic differentiated correctly on its own and gave `nan` as soon
  as a one-real-root cubic shared the batch, and `solve_quartic` inherited it through its
  resolvent cubic. The forward values are unchanged. (#4334, #4338)
- `solve_cubic`'s `D <= 0` (three-real-roots) branch no longer returns `nan` gradients for a
  repeated or near-repeated real root (#4290, #4299). It differentiates `acos(R / sqrt(-Q3))`, whose own
  derivative `-1/sqrt(1-x^2)` is unbounded at `x = +-1`; the branch condition guarantees the ratio
  lies in `[-1, 1]`, but a repeated or near-repeated root pushes it to exactly that boundary,
  where the value is correct but the derivative diverges. A double-root quartic reaches this
  through `solve_quartic`'s resolvent cubic routinely, not as an edge case: measured on 20,000
  random real-rooted quartics, 10.7% hit a non-finite gradient or a disconnected graph overall,
  35.3% for exact double-root pairs specifically. Same failure shape as the `acos`/`asin`
  boundary in `quaternion_exp_to_log`/`euler_from_quaternion` (#4007, fixed in #4228) — this is a
  different call site, not covered by that fix, using the same guard: differentiate a substituted
  safe argument, take the value from a detached copy of the real (possibly boundary) argument.
  Forward values are unchanged; re-running the same 20,000-trial search with the fix applied
  drops the failure count to 1,269 — a real, separate residual (a disconnected-graph case with
  two simultaneous double-root pairs) remains and is tracked in #4290, not fixed here.
- `solve_quartic` now validates Ferrari resolvent candidates with a dtype-aware scaled residual,
  falls back to the smallest scaled residual when no candidate passes, and keeps the Ferrari `E`
  factorization away from an ill-conditioned `R` division. Cancellation-level `R^2` is snapped to
  zero relative to the coefficient scale, and all Ferrari compute dtypes choose between the linear-
  and constant-term `E` forms by normalized coefficient error; float32 also refines the selected
  resolvent root once. The gradient-safe zero-radicand path remains intact, preventing half-precision
  non-finite or dropped roots and full-precision non-roots while preserving the public contract. (#4357)
- `solve_quartic` returns finite gradients for a pure biquadratic such as `x^4 - 16`. Its two
  `torch.clamp(..., min=0.0).sqrt()` sites do not guard the gradient they look like they guard:
  `d(sqrt)/dx` is unbounded at 0, and on torch below 2.14 `clamp` passes the incoming gradient
  through at the bound rather than zeroing it, so on the older half of kornia's supported torch
  range the backward returned `inf` and then `nan` (#4229). Both sites now substitute a safe
  radicand under the `sqrt`, as `solve_quadratic` in the same module already did. Forward values
  are unchanged. (#4339)
- `tilt_projection` preserves its documented leading batch dimensions, so `distort_points`,
  `undistort_points`, and `undistort_image` no longer fail on multi-axis batches when tilt distortion
  is applied. (#4345)
- `undistort_points_kannala_brandt` no longer collapses representable `float16` points next to the principal point
  to the origin, and the exact principal-point path now has finite autograd gradients. The zero-radius decision is
  made from the unsquared normalized coordinates, and the `float16` radius uses `float32` intermediates so its
  squared value does not underflow; nonzero radial rescaling remains epsilon-free. (#4308, #4370)
- `unproject_meshgrid` now requires camera intrinsics of shape `(B, 3, 3)`, rejecting extra
  camera axes before they can broadcast across pixel columns and reporting the caller's original shape. (#4383)
- `unproject_meshgrid` now returns the documented `(*, H, W, 3)` shape when `W` is 1. It squeezed the
  meshgrid with a bare `.squeeze()`, which drops the width axis along with the leading batch axis whenever
  `W == 1`, so `unproject_meshgrid(1, 3, K)` and `unproject_meshgrid(3, 1, K)` returned the same shape with
  different values. The grid then broadcast across a phantom width in the two public functions built on it:
  `depth_to_3d_v2` returned `(1, 3, 3, 3)` for a `W = 1` depth where `depth_to_3d` correctly returns
  `(1, 3, 1, 3)`, and `warp_frame_depth` returned a 4-pixel-wide image for a 1-pixel-wide input. Only the
  batch axis is dropped now. Output for every `W > 1` is byte-identical, and `depth_to_3d_v2` agrees with
  `depth_to_3d` at `W = 1` as it already did everywhere else. (#4278)
- `validate_bbox` and `validate_bbox3d` flatten rank-4 `(B, N, 4, 2)` / `(B, N, 8, 3)` input with `reshape`
  instead of `view`, so a non-contiguous leading-dimension stride (a transpose, a slice that drops boxes, an
  `expand`) returns a boolean as documented instead of raising `RuntimeError`. (#4174)
- `warp_affine`, `warp_perspective` and `remap` crashed on MPS for an empty destination -- a `dsize` with a
  zero dimension, or zero-sized `remap` maps -- with an internal
  `[srcBuf length] > 0 INTERNAL ASSERT FAILED ... Placeholder tensor is empty!` from PyTorch. The MPS backend
  rejects *any* zero-element `grid_sample` operand before torch 2.14, including a zero-element grid sampled
  against a non-empty source, which is what the empty-destination path built. That path now samples a connected
  1x1 stand-in and expands the result to the requested empty shape, so `grid_sample` never receives a
  zero-element operand on any backend and an empty warp costs the same whatever the non-zero side of `dsize` is.
  The empty path also resolves its output batch the way each operation's non-empty path does, so a singleton
  transform or map batch broadcasts identically and a mismatched batch is rejected rather than silently
  returning the wrong cardinality. Outputs, autograd links and the documented empty-source policy are unchanged
  on CPU and CUDA. (#4032, #4354)
- `ycbcr_to_rgb` now uses the exact inverse of the `rgb_to_ycbcr` coefficients, so an
  `rgb -> ycbcr -> rgb` round trip is lossless to floating-point precision instead of drifting
  by ~2.7e-4. The forward transform is unchanged. (#4378)

### Breaking changes

- `mean_absolute_disparity_error`, `root_mean_squared_disparity_error` and `mean_bad_pixel_error`
  now raise a kornia error naming the dtypes when either disparity map is not floating point, as
  does the new `kitti_d1_error`. Previously `reduction='mean'` already failed on integer maps with a
  torch `RuntimeError` from `mean()`, but `reduction='sum'` and `reduction='none'` returned results:
  an `int64` sum and an integer per-pixel map for the absolute and bad-pixel metrics, and `float32`
  for the root-mean-square one. Those calls now raise. The guard exists because the reductions
  accumulate in `float32` and cast back to the map's dtype, which for an integer map would truncate
  a ratio such as 0.5 to 0, and KITTI ships disparity as a `uint16` PNG, so a plausible-looking
  wrong score was the alternative. Convert integer maps with `.float()` before calling. ([#3931](https://github.com/kornia/kornia/pull/3931))
- Outputs of `warp_perspective`, `warp_affine`, `homography_warp`, `HomographyWarper`,
  `warp_image_tps`, `crop_by_transform_mat` and `shear` change under `align_corners=False`. They now
  match the `align_corners=True` result up to out-of-bounds handling, instead of carrying a spurious
  sub-pixel scale and shift. `align_corners=True` call sites are bit-identical.

  **Several of these default to `align_corners=False`, so plain calls that pass no flag change too.**
  Measured on a 4x5 `arange` image with an identity transform, `float64`, as `max|out - in|`:

  | Call, default arguments | before | after |
  |---|---|---|
  | `homography_warp(x, I, (4, 5))` | 14.25 | 1.8e-15 |
  | `HomographyWarper(4, 5)(x, I)` | 14.25 | 1.8e-15 |
  | `warp_image_tps(x, pts, kernel, affine)` | 14.25 | 1.8e-15 |
  | `ImageRegistrator("homography").warp_src_into_dst(x)`, identity model | 14.25 | 0.0 |
  | `shear(ones(1, 1, 4, 4), [[0.5, 0.0]])`, first row | `0.75, 1, 1, 1` | `1, 1, 1, 1` |

  The augmentations that default to `align_corners=False` (`RandomAffine`, `RandomPerspective`,
  `RandomShear`, `RandomTranslate`, `RandomThinPlateSpline`) sample different pixels for the same
  reason; their transform matrices are unchanged. At those defaults an identity transform is now
  actually the identity: `RandomPerspective(distortion_scale=0.)` goes from 0.734 to 6e-07 and
  `RandomThinPlateSpline(scale=0.)` from 0.734 to 1.5e-08 on a 6x8 ramp. Refs #3904, #4411. ([#3945](https://github.com/kornia/kornia/pull/3945))
- `PinholeCamera(...)` now clones its `intrinsics`, `extrinsics`, `height`, and `width` inputs instead of storing
  the caller's tensors by reference. `scale_()` and the `tx` / `ty` / `tz` setters still mutate the camera in place,
  but no longer write through to those constructor arguments; code that relied on bidirectional storage aliasing
  must mutate the camera's tensors explicitly. A camera returned by `PinholeCamerasList.get_pinhole()` is likewise
  independent of the list's backing storage. (#4264) ([#4500](https://github.com/kornia/kornia/pull/4500))
- A scalar augmentation magnitude `x` whose implied range `center ± x` reaches past the parameter's upper
  bound now raises the same `ValueError: <name> out of bounds. Expected inside (lo, hi), got tensor([...])`
  that the explicit range form already raised; it used to be clamped to the bound silently. The lower end is
  still floored at the parameter's lower bound, so `RandomContrast(1.5)` still means `[0, 2.5]`,
  `RandomSharpness(0.5)` still means `[0, 0.5]`, `RandomJPEG(50.0)` still means `[1, 100]` and
  `RandomAffine(degrees=360)` still means `[-360, 360]`. The scalar forms that now raise are
  `hue > 0.5` (`RandomHue`, `ColorJiggle`, `ColorJitter`), `brightness > 1` (`RandomBrightness`,
  `ColorJiggle`), `thresholds > 0.5` and `additions > 0.5` (`RandomSolarize`), `jpeg_quality > 50`
  (`RandomJPEG`), `direction > 1` (`RandomMotionBlur`, `RandomMotionBlur3D`), `angle > 360`
  (`RandomMotionBlur`), `degrees > 360` (`RandomRotation`, `RandomAffine`), `shear > 360` (`RandomShear`,
  `RandomAffine`), `translate > 1` (`RandomAffine`), `translate_x > 1` and `translate_y > 1`
  (`RandomTranslate`) and `times > 3` (`RandomRotation90`). The 3D `degrees`, `shears` and `angle`
  arguments go through a different reader and are unchanged. Refs #4563. ([#4569](https://github.com/kornia/kornia/pull/4569))
- Casting `RandomPlanckianJitter` to a dtype no longer changes the dtype of its output. The illuminant
  table now follows the input rather than the module, so `K.RandomPlanckianJitter(p=1.0).to(torch.float64)`
  applied to a `float32` input returns `float32` where it previously returned `float64`, and a module cast
  to the other half dtype returns the input's dtype instead of `float32`. Casting the module is no longer
  the way to keep a half-precision input's dtype; it is preserved by default. Refs #4574. ([#4578](https://github.com/kornia/kornia/pull/4578))
- `RandomMotionBlur` and `MotionBlurGenerator` now raise `ValueError` at construction for a reversed `kernel_size` range such as `(20, 3)`. Previously a reversed range was silently sampled as if it were ordered: `(20, 3)` drew the odd sizes `3` to `19`. ([#4610](https://github.com/kornia/kornia/pull/4610))
- `StereoCamera` now rejects a rectified pair whose `tx * fx` is not strictly negative for every rig in the batch.
  Its sign guard used `torch.all`, so a batch holding one pair with its cameras swapped was accepted and silently
  reprojected that pair behind the camera, and a zero baseline passed too, collapsing `Q` so that every disparity
  reprojected to the origin. A zero baseline now raises with its own message naming `tx`; an empty batch is still
  accepted, and valid rigs build the same `Q` as before. ([#4616](https://github.com/kornia/kornia/pull/4616))
- `AugmentationSequential(data_keys=None)` now matches dictionary data-key names exactly or before an
  underscore/hyphen suffix, preserving coordinate formats for keys such as `bbox_xyxy_2` and leaving the
  caller's dictionary intact. Formerly augmented plural names (`images`, `masks`, `labels`, `bboxes`, `inputs`),
  raw suffixes (`image2`, `keypoints2`), camelCase (`imageLeft`), dot-separated names (`image.2`), and unrelated
  prefixes (`imagenet_id`, `maskrcnn_boxes`, `labelled_image`, `inputsize`) are now returned unchanged as
  metadata without a warning. Rename augmentation inputs to recognized names such as `image`, `mask_2`,
  `label_2`, `bbox_2` or `input_2`. `class` and any key beginning with `class_` or `class-`, for example
  `class_id` or `class_weights`, formerly metadata, now route to labels and can raise for unsupported label-
  changing mix augmentations; rename metadata that must remain untouched. Unrecognized `keypoint` remains
  metadata in dict mode, whereas positional mode rejects it; use `keypoints` for augmentation. A coordinate-box
  suffix still needs `_` or `-`: `bbox_xyxy2` and `bbox_xywh2` match `bbox` and require vertex boxes. ([#4619](https://github.com/kornia/kornia/pull/4619))
- A scalar augmentation magnitude of `inf` — or a finite magnitude too large to represent in the
  working dtype, which converts to `inf` — now raises `ValueError: If <name> is a single number, it
  must be finite` at construction. Previously `RandomSharpness`, `RandomContrast`, `RandomSaturation`,
  `ColorJitter` (`contrast`, `saturation`) and `ColorJiggle` (`contrast`, `saturation`) accepted such a
  magnitude and the resulting infinite draw range produced `NaN` pixels at forward time. A scalar `nan`
  was already rejected by the generic range check on every reachable path and now gets the same named
  message instead. `-inf` still reports "must be non negative" from the earlier check, and the explicit
  two-element range form (e.g. `RandomSharpness((0.0, float("inf")))`) is unchanged and still accepted —
  tracked in #4635. Fixes #4621. ([#4625](https://github.com/kornia/kornia/pull/4625))
- The 3D augmentations now reject an angle range wider than one turn at construction, as the 2D ones do: `RandomRotation3D`, `RandomAffine3D` (`degrees` and `shears`) and `RandomMotionBlur3D` raise `ValueError` for a scalar or explicit range past `±360`, where they used to sample from it. ([#4644](https://github.com/kornia/kornia/pull/4644))
- `box_blur` and `BoxBlur` now default to `separable=True` instead of `False`, using two one-dimensional averaging passes. Both dense and separable floating-point paths use average pooling outside autocast (except where oneDNN's CPU convolution is faster), so results can differ by roundoff. `BoxBlur` no longer stores derived `kernel`, `kernel_x`, or `kernel_y` buffers: old checkpoints still load strictly, but their kernel values are ignored. Customizing these buffers no longer changes the filter; use `filter2d` or `filter2d_separable` for custom weights. ([#4647](https://github.com/kornia/kornia/pull/4647))
- `AugmentationSequential` no longer converts every mask output to one shared dtype. Previously all mask outputs of a
  call were cast to the dtype of the last mask argument, so a boolean mask passed before a `float32` mask came back as
  `float32`; it now comes back as `bool`, like every other mask that keeps its own argument's dtype. Code that did
  arithmetic on such an output (for example `1 - mask`, which raises for `bool` tensors) must convert it explicitly,
  e.g. `mask.float()`. ([#4665](https://github.com/kornia/kornia/pull/4665))
- An explicit augmentation range containing `inf` or `nan` — for example
  `RandomSharpness((0.0, float("inf")))`, `ColorJitter(contrast=(0.0, float("inf")))` or
  `RandomAffine(0., scale=(0., float("inf")))` — now raises
  `ValueError: If <name> is a range, it must be finite` at construction. Previously an `inf` endpoint
  was accepted whenever the parameter's domain was itself unbounded (the default `bounds=(0, inf)`),
  because the joint check then compares `inf >= inf`; the resulting infinite draw range produced `NaN`
  pixels at forward time, the same defect #4621 reported for the scalar form. With a finite domain such
  a range was already rejected. A `nan` endpoint was already rejected too, but reported as
  "out of bounds", naming the domain rather than the endpoint, and now gets the same named message.
  Finiteness is required of the range, not of `bounds`: an unbounded domain stays legal. This completes
  the scalar-form fix in #4625. Fixes #4635. ([#4668](https://github.com/kornia/kornia/pull/4668))
- `median_blur` and `MedianBlur` take a `border_type` (`'constant'`, `'reflect'`, `'replicate'` or `'circular'`) and
  default to `'reflect'`, like `filter2d`, `box_blur` and `gaussian_blur2d` (#4670). Previously the window was
  zero-padded and there was no `border_type`, so a constant image came back 0 in a triangle at each corner (12 of the
  121 pixels of an 11×11 image with a 5×5 kernel). Now a constant image is unchanged. `RandomMedianBlur`, which has no
  `border_type`, follows the new default. Reflect padding needs each side to be longer than the kernel radius, so an
  image with a side of at most `k // 2` pixels, which used to be filtered, now raises: torch's padding error from
  `median_blur`, and a `ValueError` naming the class and the shape from `RandomMedianBlur`. On a torch without a
  float16 CPU reflection-pad kernel, such as 2.5.1, a float16 CPU input now raises, as it does in the other
  reflect-padded filters. `border_type='constant'` reproduces the old output exactly. ([#4673](https://github.com/kornia/kornia/pull/4673))
- `spatial_gradient(..., order=2)` and `spatial_gradient3d(..., order=2)` return second derivatives with the
  standard sign: on `x**2` the `dxx` channel is positive and on `x*y` the `dxy` channel is positive, matching the
  OpenCV `Sobel(dx=2)` and `Sobel(dx=1, dy=1)` kernels. Previously `mode="sobel"` returned `-dxx`, `-dxy` and `-dyy`,
  the 2D `mode="diff"` returned `-dxy` next to a correctly signed `dxx`/`dyy`, and the 3D `mode="diff"` returned
  `-dyz` and `-dxz` next to a correctly signed `dxy`. Code that compensated for the old signs, for example `-dxx` as a
  Sobel Laplacian, now gets the opposite result without a warning. The exported kernel getters
  `get_spatial_gradient_kernel2d("sobel", 2)` and `get_spatial_gradient_kernel2d("diff", 2)` return the negated
  kernels accordingly, while `get_spatial_gradient_kernel3d("diff", 2)` is unchanged because the fix removes the
  flip in `spatial_gradient3d` instead of editing the kernel. `hessian_response` is unchanged. The docstrings now
  list the channel layout, `(dxx, dxy, dyy)` in 2D and `(dxx, dyy, dzz, dxy, dyz, dxz)` in 3D. ([#4683](https://github.com/kornia/kornia/pull/4683))
- `AugmentationSequential` now rejects rank-4 image input when used with a 3D augmentation. Previously, rank-4 input was promoted as ``(C, D, H, W)``; this is ambiguous with the container's ``(B, C, H, W)`` image layout. To pass a single 3D volume explicitly, use ``unsqueeze(0)`` to add the batch dimension and provide input as ``(B, C, D, H, W)``. Bare 3D augmentations retain their rank-3 and rank-4 input promotion behavior. ([#4697](https://github.com/kornia/kornia/pull/4697))
- `CenterCrop` in the default `cropping_mode="slice"` now returns a copy of the cropped region. Previously it returned a
  view that shared storage with the input: writing to the output changed the input at the crop offset, and writing to the
  input after the call changed the output. Code that relied on either direction of that aliasing must now write to the
  tensor it wants changed. The crop now allocates one crop-sized tensor per call, as `RandomCrop` and `RandomResizedCrop`
  already do in slice mode, and no longer keeps the whole input alive through the output. Fixes #4413. ([#4727](https://github.com/kornia/kornia/pull/4727))
- The `kornia.morphology` functions (`dilation`, `erosion`, `opening`, `closing`, `gradient`, `top_hat`, `bottom_hat`) now default to `engine="auto"` instead of `"unfold"`. `"auto"` runs `"unfold"` on CUDA; off CUDA it runs the exact `"shift"` engine, except that a CPU call which computes in float32 or float64 (the image dtype, or the wider dtype a kernel or structuring element promotes it to) and records a backward graph keeps `"unfold"`. For finite inputs the forward output remains equal to `"unfold"` (`torch.equal`), including mixed-dtype kernels and non-flat structuring elements; only the sign of a zero can differ, because a backend's `max`/`min` may return either tied operand, and because forward-only `"shift"` skips adding the zero offsets of a call without `structuring_element`, so its `dilation` keeps a `-0.0` maximum that `"unfold"` returns as `+0.0`. Gradients change only where `"shift"` is now selected: CPU float16/bfloat16, MPS, and forward-mode AD (`torch.func.jvp`) off CUDA, which records no backward graph. Previously a window with tied maxima sent its whole gradient to one argmax; `"shift"` splits it along its traversal order: half goes to the last tied position it visits, a quarter to the one before, and so on, so pixels that received exactly zero can now receive a fraction (the total is unchanged, and in a large flat window the earliest positions' share underflows to zero). Pass `engine="unfold"` to keep the previous gradient behaviour. ([#4728](https://github.com/kornia/kornia/pull/4728))
- `kornia.morphology.dilation` (and `closing`, `gradient`, `top_hat`, `bottom_hat`, `opening`, which call it) now
  anchors the reflected structuring element at `origin`. It used to reflect the kernel values but pad with the
  unreflected origin, so an even-sized kernel (for example `torch.ones(2, 2)`) shifted the result by one pixel
  toward the bottom-right relative to `scipy.ndimage.grey_dilation` and `skimage.morphology.dilation`, and a custom
  `origin` moved the window in the opposite direction to the one `erosion` uses. Odd kernels with the default origin
  are unchanged. ([#4737](https://github.com/kornia/kornia/pull/4737))
- Morphology operations now raise a `ValueError` naming the argument for an unknown `border_type` (PyTorch
  raised `NotImplementedError: Unrecognised padding mode`) and for a `structuring_element` whose shape differs
  from `kernel`'s (PyTorch raised an internal `IndexError` or `RuntimeError`; only an element with extra trailing
  size-1 axes, such as `(k_h, k_w, 1)`, used to run, and only with `engine="convolution"`). On a floating-point
  image, a `bool` or integer `kernel` is now a plain membership mask and gives exactly the result and dtype of
  the same kernel in the image's dtype. Before, a `uint8` or `int8` kernel raised an overflow `RuntimeError`, and
  a `bool` kernel stored its excluded cells as `True` instead of `-max_val`: `dilation` returned wrong values on
  every engine, `erosion` did so with `engine="convolution"` and raised on the other engines, and the five
  operations built on them inherited this. Non-float images behave as before. ([#4744](https://github.com/kornia/kornia/pull/4744))
- `adjust_gamma` and `AdjustGamma` now raise `ValueError` when `gamma` or `gain` has more dimensions than the image, the same error `adjust_brightness`, `adjust_contrast`, `adjust_saturation` and `adjust_hue` raise for such a `factor` (#4720). Previously the over-ranked factor silently broadcast the image up to a larger output shape: a `(3, 4, 4)` image with a `(2, 1, 1, 1)` `gamma` returned a `(2, 3, 4, 4)` tensor. ([#4755](https://github.com/kornia/kornia/pull/4755))
- `crop_by_indices` now honors `shape_compensation` when every box in the batch is identical. With a `shape_compensation` other than `"resize"` and a slice that does not match `size`, it used to resize the slice to `size` whenever all boxes in the batch coincided (always for a batch of one) and zero-pad or trim it otherwise; it now zero-pads or trims it in both cases, so a row's output no longer depends on the other rows in the batch. `RandomMosaic(cropping_mode="slice")` crops this way (`shape_compensation="F.pad"`): for a batch of one, or any batch whose crop boxes coincide, a crop that does not match `output_size` (for example because `output_size` differs from the input size) used to be resized to `output_size`, out of register with the transformed boxes; it is now zero-padded or trimmed at the bottom and right, as batches with differing boxes already were, so fixed-seed outputs change. ([#4756](https://github.com/kornia/kornia/pull/4756))
- `ColorJitter(order=...)` and `ColorJiggle(order=...)` now raise `ValueError` for a repeated index. They used to
  accept it and apply that adjustment once per occurrence with the same drawn factor, so `order=(0, 0, 1)` applied
  brightness twice. (#4719) ([#4760](https://github.com/kornia/kornia/pull/4760))
- `TrivialAugment`, and `PolicySequential` when used directly, now sample each operation through its
  `OperationBase` wrapper, so a policy entry means what it means under `RandAugment`. They used to call the
  wrapped augmentation's own sampler, which skipped the wrapper's magnitude mapping: `("shear_x", -0.3, 0.3)`
  sheared by at most `0.3` degrees instead of up to `54`, and the symmetric ops (`rotate`, `shear_x`,
  `shear_y`, `translate_x`, `translate_y`) never drew a negative magnitude. A `PolicySequential` built from
  operations with an `initial_magnitude` used to ignore it; it now uses that magnitude, as the wrapper does,
  and the magnitude receives a gradient. For a fixed seed, `TrivialAugment` now draws different magnitudes,
  and a symmetric op draws its random signs from the global generator, so every later random draw in the
  pipeline changes as well. `AutoAugment` samples as before, up to float32 rounding of the shear values: its
  shear bins were pre-scaled to degrees to work around the skipped mapping and are now fractions, which the
  mapping scales once. Fixes #4441. ([#4761](https://github.com/kornia/kornia/pull/4761))
- `ImageSequential(if_unsupported_ops=...)` now takes effect. `inverse` used to skip a plain `nn.Module` member
  (a `kornia.filters` blur, for example) under every value of the argument, returning the input with that module
  still applied and without an error. With the default `'raise'` it now raises `NotImplementedError` naming the
  module, before anything is inverted; `'skip'` keeps the previous behavior. Code that inverted through a plain
  module must now pass `if_unsupported_ops='skip'`. A value other than `'raise'` or `'skip'` raises `ValueError`
  at construction, where it was stored silently. `VideoSequential` inverts through the same path and now takes
  the same argument. `AugmentationSequential` still leaves a plain module child applied, also when it is
  nested in another container, but an `ImageSequential` or `VideoSequential` nested in it follows its own
  setting and so now raises by default as well. `PatchSequential` is unaffected. Fixes #4423. ([#4764](https://github.com/kornia/kornia/pull/4764))
- `spatial_gradient(..., order=2, normalized=True)` and `SpatialGradient(order=2)` now scale every second order
  kernel by its response to the quadratic it estimates, so `dxx`, `dxy` and `dyy` come out in the same units: on
  `x**2 / 2`, `x * y` and `y**2 / 2` each channel returns 1 in both modes. Previously each kernel was divided by its
  own absolute sum, which gave the mixed channel a scale of its own: `mode="sobel"` returned `dxy` too large by
  `64 / 36`, `mode="diff"` returned `dxx` and `dyy` too small by 4. `order=1` and `normalized=False` are unchanged.

  `hessian_response` and `BlobHessian` use the normalized default, so their values change: the determinant is now
  `fxx * fyy - fxy**2` instead of `fxx * fyy - (64 / 36)**2 * fxy**2` (sobel) or `fxx * fyy / 16 - fxy**2` (diff).
  At the centre of a 9x9 grid:

  | surface (true determinant) | sobel before | sobel after | diff before | diff after |
  |---|---|---|---|---|
  | `x**2 + y**2` (4) | 4.0 | 4.0 | 0.25 | 4.0 |
  | `x * y` (-1) | -3.1605 | -1.0 | -1.0 | -1.0 |
  | `(x + y)**2 / 2` (0) | -2.1605 | 0.0 | -0.9375 | 0.0 |

  With the default `grads_mode="sobel"` responses without a mixed term keep their value and the `fxy**2` term
  shrinks; with `grads_mode="diff"` the `fxx * fyy` term grows 16 times. The detections of `ScaleSpaceDetector`
  (whose default response is `BlobHessian`), `HesAffNetHardNet`, and `MultiResolutionDetector` with `BlobHessian`
  can therefore change. Refs #4710. ([#4767](https://github.com/kornia/kornia/pull/4767))
- `ellipse_to_laf` now returns the LAF of the ellipse it is given. For an Oxford ellipse `(x, y, a, b, c)` the
  returned upright, lower-triangular `A` satisfies `A @ A.T == inverse([[a, b], [b, c]])`, so the LAF maps the unit
  circle onto `a x**2 + 2 b x y + c y**2 = 1` and `ellipse_to_laf` of the ellipse implied by a LAF `A0` returns
  `make_upright(A0)`. Previously the function took the triangular square root of the lower-triangular matrix
  `[[a, 0], [b, c]]` instead of the symmetric `[[a, b], [b, c]]`, so whenever `b != 0` the returned region was
  under-tilted: for `(a, b, c) = (1, 0.5, 1)` the region it described was `[[1.0625, 0.25], [0.25, 1]]`, for
  `(2, -0.9, 0.5)` it was `[[2.18, -0.3], [-0.3, 0.5]]`. Valid ellipses with `b == 0` give bit-identical results.

  `LAFAffineShapeEstimator` feeds its second moment matrix through this function, so its output changes for every
  patch whose dominant gradient direction is not axis-aligned. On an anisotropic Gaussian blob rotated by 45 degrees
  the estimated region's major axis is now at 45.0 degrees with axis ratio 4.16; before it was at 51.3 degrees
  with axis ratio 1.56. Patches whose mixed moment is exactly zero are unchanged.

  Inputs that are not positive definite (`a`, `c` or `a * c - b * b` not positive) describe no ellipse. Previously a
  negative `a` or `c` was replaced by its absolute value and `b` did not enter the diagonal, so a negative diagonal
  entry or `b * b >= a * c` still produced a finite LAF; now such an input gives `inf` or `nan` entries, or huge finite
  ones where rounding leaves `a - b * b / c` barely positive. A `c == 0` input now also has `nan` in `A[0, 0]`, which
  used to be `1 / sqrt(a)`. Screen results with `laf_is_valid`. Refs #4684. ([#4768](https://github.com/kornia/kornia/pull/4768))
- `ScalePyramid` with an `init_sigma` below the assumed input blur (`0.5`, or `1.0` with
  `double_image=True`) now builds and labels every octave from the first level's real blur, so its
  images and `sigmas` equal those of a pyramid built with `init_sigma` set to the input blur.
  Previously the level-to-level blurs and `sigmas` counted from `init_sigma`: the later levels were
  blurred less, `sigmas` did not match their blur (octave 0 labelled two different levels with the same
  sigma), and `init_sigma=0` returned NaN images. An `init_sigma` at or above the input blur,
  including the default `1.6`, is unchanged. ([#4812](https://github.com/kornia/kornia/pull/4812))
- `adjust_contrast_with_mean_subtraction`, `AdjustContrastWithMeanSubtraction` and the contrast step of `ColorJitter`
  now blend an input that is not three-channel with each image's own mean over its channels and pixels. They used to
  blend it with the mean of the whole tensor, batch and any other leading dimensions included, so an image's result
  depended on the other images in the batch. Batched outputs for such inputs change; a single image, `(C, H, W)` or
  a batch of one, and three-channel inputs give the same results as before. Fixes #4806. ([#4816](https://github.com/kornia/kornia/pull/4816))
- Morphology no longer uses the finite `max_val` sentinel for floating-point images. Before, a window with no kernel cell inside the image under `border_type="geodesic"` returned a finite value built from `max_val` (for example `-9999.0` from `dilation` with the default `max_val=1e4`), and a cell the kernel excludes still contributed its pixel minus `max_val`, so data of magnitude near `max_val` gave wrong results (`dilation` of `[0, 5e4, 0]` by `[[1, 0, 1]]` returned `40000` in the middle instead of `0`). Now an empty window returns `-inf` from `dilation` and `+inf` from `erosion`, as scipy and scikit-image do; `opening`, `closing`, `gradient`, `top_hat` and `bottom_hat` inherit those infinities; and `max_val` has no effect on floating-point images. ([#4818](https://github.com/kornia/kornia/pull/4818))
- `RandomResizedCrop` now follows torchvision's `get_params` in two places. A candidate crop may equal the input's
  width or height; it used to have to be strictly smaller on both axes. When none of the ten candidates fits, the
  fallback compares the input's width/height with both `min(ratio)` and `max(ratio)`: an input already in range is
  kept whole, a narrower one keeps its full width and a wider one its full height. It used to compare height/width with
  `min(ratio)` only, which could return a crop outside both `scale` and `ratio`. `scale=(1.0, 1.0)` therefore keeps an
  input whose width/height is within `ratio` whole; with the default `ratio` it used to crop every square or portrait
  input and most landscape ones (an 8x6 input gave a 4x6 crop, a 7x8 input a 6x8 crop). Like torchvision, it also
  accepts a candidate rounded a pixel short on the longer side: an 8x6 input gives 7x6 in about 18% of draws, and a
  6x8 input, which used to be kept whole every time, gives 6x7 as often. Seeded crops change for configurations that
  reach the fallback (a `scale` close to `(1.0, 1.0)`), and for draws where a candidate exactly matches the input width
  or height: about 1% of draws with the default `scale` and `ratio` on 224x224 or 256x192 inputs, more on small images
  (7% on 32x32). The fallback crop is still placed at a random position, where torchvision centres it. Fixes #4814. ([#4823](https://github.com/kornia/kornia/pull/4823))
- `spatial_gradient3d(..., order=2)`, `SpatialGradient3d(order=2)` and the kernel returned by
  `get_spatial_gradient_kernel3d("diff", 2)` (built by `kornia.filters.kernels.get_diff_kernel3d_2nd_order`) now scale
  the mixed `dxy`, `dyz` and `dxz` kernels by 1/4, so every channel is exact on quadratic volumes: `dxy` returns 1 on
  `x * y`, as `dxx` returns 2 on `x**2`. Previously the mixed kernels were `[[1, 0, -1], [0, 0, 0], [-1, 0, 1]]` without
  the 1/4 factor, so `dxy`, `dyz` and `dxz` came out four times too large relative to `dxx`, `dyy` and `dzz`. `order=1`
  is unchanged. ([#4827](https://github.com/kornia/kornia/pull/4827))
- `solve_pnp_dlt` now applies `weights` to the point normalization and the degeneracy check as well as to the
  linear system, so a zero weight is the same as removing the point. The weights used to scale only the rows of
  the linear system: a zero-weight point still moved the normalization and counted in the degeneracy check, so
  coplanar points plus zero-weight points off the plane passed the check and returned a wrong pose (#4799). That
  input now raises the check's `AssertionError`, as do off-plane points whose weights are too small to lift the
  weighted points off the plane (for example `1e-8`). Fewer than 6 nonzero weights in a batch element now raise
  `BaseError`, as fewer than 6 points do; they used to return an arbitrary pose, or NaN for an all-zero row.
  Non-uniform weights on noisy data now return a different pose. `weights=None` gives the same result as before,
  uniform weights give it up to rounding, and only the weights' magnitudes relative to each other matter. ([#4829](https://github.com/kornia/kornia/pull/4829))
- `RandomRain` now paints a drop of height `h` and width `w` as a contiguous line of `max(h, |w|)` pixels covering
  exactly `h` rows and `max(|w|, 1)` columns. Previously a drop whose larger size was at least 2 spanned `h + 1` rows
  and `|w| + 1` columns with one step skipped, so `drop_height=5, drop_width=0` painted rows `[0, 1, 2, 3, 5]`. The
  random draws are unchanged, but the admissible start region grew to match (by one row, and by one column unless the
  width is 0), so the same seed can now place such drops at different positions. Single-pixel drops are unchanged. ([#4836](https://github.com/kornia/kornia/pull/4836))
- `find_essential` and `run_5point` recover the pose again for the roughly half of all samples whose
  degree-10 polynomial has a negative leading coefficient. Since #3458, first released in 0.8.3, that
  coefficient was clamped with `clamp_min(1e-8)`, which turns a negative coefficient into `+1e-8`, so
  the companion matrix had the wrong sign and scale and none of the 10 candidates was the true
  essential matrix. On noise-free scenes the pose was recovered for 992 of 2000 samples at float64 and
  880 at float32; it is now 1979 and 1767. At float32 with more than five points, the input gradients
  of those samples were also NaN, and now they are finite. Only an exactly zero coefficient is replaced
  now, as before #3458. Samples whose coefficient is at least `1e-8` are unchanged bit for bit; the
  candidates of any other sample can change, including degenerate ones whose coefficient is positive
  but below `1e-8`. Fixes #4847. ([#4848](https://github.com/kornia/kornia/pull/4848))
- `kornia.morphology.dilation`, `erosion`, and the functions built on them (`opening`, `closing`, `gradient`, `top_hat`, `bottom_hat`) now compute `engine="convolution"` in the same promoted dtype as `engine="unfold"` and `engine="shift"`. Previously, `engine="convolution"` returned the image's own dtype for a mismatched-dtype kernel or `structuring_element` (for example, a float16 image with a float32 kernel returned float16), while the other two engines promoted to the wider dtype (float32) — so switching `engine` silently changed the result dtype. All three engines now agree. ([#4882](https://github.com/kornia/kornia/pull/4882))
- `RANSAC` now returns a one-dimensional `(N,)` boolean inlier mask when no model is found, matching successful
  calls. Previously, the no-model mask had shape `(N, 1)` and could not be used to index the input keypoints. ([#4885](https://github.com/kornia/kornia/pull/4885))
- `find_essential` and `run_5point` now return `NaN` for all 10 candidates of a sample whose degree-10
  polynomial has an exactly zero leading coefficient, which data with no parallax, such as the same
  points in both images, gives for about half of all samples. Such a polynomial has a lower degree.
  Before, the solver divided by `1e-8` in place of the zero and returned up to 10 candidates from the
  eigenvalues of the resulting companion matrix. None of them satisfied the essential-matrix
  constraints, and for some samples the backward raised `torch._C._LinAlgError` in the backward of
  `torch.linalg.eigvals`. Now the sample returns `NaN` for all 10 candidates and its backward
  completes. Samples with a nonzero leading coefficient are unchanged bit for bit. Fixes #4831. ([#4888](https://github.com/kornia/kornia/pull/4888))
- `ColorJitter` now skips its brightness step when every factor is the multiplier's neutral `1`, instead of `0`
  (#4785). Before, a batch whose brightness factors were all `0` came back unchanged while the same sample in a mixed
  batch came back black; now a factor of `0` gives a black image whatever the rest of the batch drew. Before, the
  factor `1` drawn by the default `brightness=0.0` still ran that step, which clamps into `[0, 1]`: an out-of-range
  input was clamped before any later step, and on torch releases whose `clamp` has a zero gradient at its bounds, such
  as 2.14, input pixels at exactly `0` or `1` got no gradient. Now the step is skipped; outputs for inputs in `[0, 1]`
  are unchanged, and the contrast and saturation steps still clamp when they run. With every step neutral,
  `ColorJitter` and `ColorJiggle` now return a copy of the input instead of the input tensor itself. ([#4893](https://github.com/kornia/kornia/pull/4893))
- A mask override with `align_corners=None` in `AugmentationSequential(extra_args=...)` now means the module's own
  `align_corners` in `cropping_mode="slice"` as well as in `"resample"`. Slice mode used to keep `None`, which
  `interpolate` reads as `False`, so with `RandomResizedCrop`'s default `align_corners=True` a bilinear or bicubic
  mask was resized on a different grid from its image and no longer lined up with it. Outputs change only for that
  combination: slice-mode `RandomResizedCrop` with a bilinear or bicubic mask override, `align_corners=None`, and
  `align_corners=True` on the module. Nearest masks, `align_corners=False` and resample mode are unchanged, as are
  `RandomCrop` and `CenterCrop`. Fixes #4854. ([#4894](https://github.com/kornia/kornia/pull/4894))
- `RANSAC("homography_from_linesegments")` now compares the squared mean endpoint-to-line distance with `inl_th**2`, so
  `inl_th` is in coordinate units, as for the point models. It used to compare the unsquared algebraic residual, which
  is that distance multiplied by the target segment's length, so each segment had a different effective threshold;
  re-tune `inl_th` for this model. A zero-length target segment is an outlier. `RANSAC.max_samples_by_conf` returns
  `sys.maxsize` instead of `1` when no finite bound exists (fewer inliers than the sample size, or `conf >= 1`), and
  rounds up instead of truncating; `1` used to stop estimation after the first batch. `RANSAC` now raises `ValueError`
  for a nonpositive or nonfinite `inl_th`, a nonpositive `batch_size` or `max_iter`, a negative `max_lo_iters`, or a
  `confidence` outside `(0, 1]`. These used to be accepted: a negative `inl_th` acted as its absolute value, since it is
  squared, zero counted only exact matches and NaN found no model; a zero `batch_size` or `max_iter` returned the failure
  result without sampling; a negative `max_lo_iters` disabled local optimization; and a `confidence` of 0 or above 1
  stopped after the first batch. An unsupported `score_type` now raises at construction rather than on the first call.
  `confidence=1` now runs the full budget; it used to stop after the first batch. The CPU sampler changed, so a fixed
  `seed` draws different, still reproducible, samples than earlier releases. ([#4902](https://github.com/kornia/kornia/pull/4902))
- `find_fundamental` (`method="8POINT"`), `find_homography_dlt` and `find_homography_lines_dlt` now compute the
  Hartley normalisation with the correspondence weights, as the linear system already did, so a zero-weight
  correspondence (or segment) no longer changes the result (#4875, #4890). Before, the normalisation used every
  point unweighted, and on noisy data a far correspondence given weight 0 still moved the estimate: by 97% of `F` in
  the #4875 example. Non-uniform weights on noisy data now give a different result, and so does
  `find_homography_dlt_iterated`, which re-weights after its first solve, and with it the homography polish in
  `RANSAC`. `weights=None` gives the same result as before and uniform weights give it up to rounding; negative
  weights count as zero in the normalisation, and a batch element whose weights are all zero keeps the unweighted one. ([#4908](https://github.com/kornia/kornia/pull/4908))
- `normalize_transformation`, and with it `find_fundamental`, now divides by the last entry itself instead of
  `M[2, 2] + 1e-8`, so the returned last entry is exactly 1 (#4874). Before, it was `M[2, 2] / (M[2, 2] + 1e-8)`: 0.990
  for `M[2, 2] = 1e-6` and 101 for `-1.01e-8`, and the backward returned NaN for a matrix whose last entry was exactly
  `-1e-8`. `find_homography_dlt` and `find_homography_lines_dlt` now end with `normalize_transformation` as well. They
  used to divide by `H[2, 2] + 1e-8` without a guard, so a homography whose last entry was at most `1e-8` in magnitude
  came back multiplied by at least `5e7`, with a last entry anywhere below `0.5`; it is now returned at its
  unnormalised scale. At the usual scale, `|H[2, 2]|` of order 1, the result changes by about `1e-8`, relative. ([#4911](https://github.com/kornia/kornia/pull/4911))
- `find_essential` and `run_5point` now return `NaN` for all 10 candidates of a five-point sample whose
  degree-10 polynomial has no real root, the same as the complex slots of any other sample, so an
  `isfinite` filter and `RANSAC` drop it. Before, such a sample returned ten 3x3 identity matrices,
  which are finite but not essential matrices. Code that detected a failed sample by comparing its
  candidates with the identity should test `torch.isnan(E).flatten(-3).all(-1)` instead. The samples
  that the fixes for #4765, #4820 and #4831 route to the same path (a singular elimination matrix, a
  non-finite companion matrix, a zero leading coefficient) return `NaN` for all 10 candidates as well.
  Every other sample, and every gradient taken through `nan_to_num`, is unchanged bit for bit. Fixes #4883. ([#4922](https://github.com/kornia/kornia/pull/4922))
- `So3.log`, and with it `Se3.log`, now returns the principal logarithm, the rotation vector with `|omega| <= pi`, for
  every element (#4925). Before, an element stored with quaternion real part `w < 0`, which includes every `exp(v)` with
  `|v| > pi` and half of `Se3.random`, returned the vector of length `2 pi - theta` about the negated axis:
  `So3.from_wxyz(-q).log()` differed from `So3.from_wxyz(q).log()` for the same rotation, `|So3.exp(4 * axis).log()|` was
  `4.0` and is now `2 pi - 4 = 2.2832`, and `Se3.log` of a 1e-6 rad rotation stored with `w < 0` returned a translation
  part of size `6.4e6` where the right one is `3.74`.
  `So3.log` is now also invariant to the scale of the stored quaternion: a non-unit `q` logs to the rotation of
  `q / |q|` (`So3(Quaternion(torch.ones(1, 4))).log()` was `[0, 0, 0]` and is now `[1.2092, 1.2092, 1.2092]`, the
  120-degree rotation about `(1, 1, 1) / sqrt(3)`; #4942 tracks the rest of `So3` for non-unit input), and its gradient
  with respect to the raw quaternion no longer has a component along `q`. The component tangent to the unit sphere, which
  is all that reaches a caller who goes through `So3.exp`, is unchanged. ([#4941](https://github.com/kornia/kornia/pull/4941))
- `NamedPose.from_rt` keeps the batch of a rotation tensor. A `(B, 3, 3)` or `(B, 2, 2)` rotation tensor with a `(B, 3)` or `(B, 2)` translation now builds a `(B, 4, 4)` or `(B, 3, 3)` pose, as `NamedPose.from_matrix` and the `So3`/`So2` rotation path do. Previously `B > 1` raised a torch `RuntimeError`, and `B = 1` returned an unbatched `(4, 4)` or `(3, 3)` pose with a `(3,)` or `(2,)` translation; it now returns `(1, 4, 4)` or `(1, 3, 3)` with a `(1, 3)` or `(1, 2)` translation. The translation must now have exactly the rotation's batch shape, otherwise `from_rt` raises a `ValueError`: a `(1, D, D)` rotation with a `(D,)` translation, and a `(D, D)` rotation with a `(1, D)` translation, used to return an unbatched pose and now raise. An unbatched `(D, D)` rotation with a `(D,)` translation is unchanged. (#4950) ([#4950](https://github.com/kornia/kornia/pull/4950))
- `Quaternion.__pow__` now keeps the norm, `q**t == ‖q‖**t (cos tθ + n sin tθ)`, so `q**1 == q`, `q**2 == q * q` and
  `q**-1 == q.inv()` for non-unit quaternions. Previously it returned the power of the unit quaternion `q/‖q‖` for every
  `q`: `Quaternion([2, 0, 0, 0])**2` was `[1, 0, 0, 0]` and is now `[4, 0, 0, 0]`. Call `q.normalize() ** t` for the old
  result. Unit-quaternion results are unchanged up to rounding, and `slerp`, which normalizes its inputs, is not
  affected. The gradient of `q**t` is now finite for a nonzero quaternion with a zero vector part, such as the identity,
  where it used to be `nan`. ([#4951](https://github.com/kornia/kornia/pull/4951))
- `RANSAC` gains `batch_size="auto"`, now the default, and `max_samples`. Before, every device drew batches of 2048
  hypotheses and `max_iter` counted them. Now the budget is `max_samples` if given, otherwise `2048 * max_iter` as
  before, and the batch is chosen per call: batches of up to 8192 homography or 2048 epipolar hypotheses on CUDA and
  MPS, fewer once the `batch x N` residual matrix would exceed `2**27` entries, and on CPU 256 to 2048 hypotheses for
  homographies or 128 to 512 for the epipolar models, fewer for more correspondences (`RANSAC.resolve_batch_size`).
  Other devices retain 2048-sample batches.
  Pass an integer `batch_size` to keep a fixed batch. `max_samples` must be an integer; a float raises `ValueError`.
  Results for a given seed depend on the batch, so default results change on CUDA, MPS and on CPU inputs large
  enough to lower the batch below 2048, and seeded local optimization with `lo_sample_size` draws from a stream offset
  by the sample budget instead of `max_iter`, so those results change too. `find_homography_dlt_iterated` and
  `find_homography_lines_dlt_iterated` re-weighted each correspondence with `exp(-e / (2 * soft_inl_th**2))` of the
  unsquared error `e`, a kernel in which `soft_inl_th` had no pixel meaning (#4870). They now use the Gaussian
  `exp(-e**2 / (2 * soft_inl_th**2))` of the error in pixels, for line segments the perpendicular distance of the
  mapped endpoints from the image-2 line, so `soft_inl_th` is a standard deviation in pixels, and `RANSAC`'s homography
  polishers set it to `inl_th`. Polished homographies change accordingly. ([#4979](https://github.com/kornia/kornia/pull/4979))
- `kornia.geometry.pose` now defines `__all__ = ["NamedPose"]`, so the stdlib module `uuid`, the `__future__` feature
  `annotations`, `torch` and the validation helper `check_matrix_shape` (renamed to `_check_matrix_shape`) are no longer
  star-imported into `kornia.geometry`. `kornia.geometry.Quaternion` stays available through an explicit re-export in
  `kornia/geometry/__init__.py`, so it no longer depends on the leak. None of the removed names is documented, and per
  `docs/source/get-started/stability.rst` undocumented helpers are private regardless of importability. The names are
  dropped from `tests/api_surface.json` in this change. `kornia.geometry.torch` remains on the surface until
  `kornia/geometry/homography.py` also defines an `__all__`. (#4939) ([#4986](https://github.com/kornia/kornia/pull/4986))
- `RandomGaussianIllumination` places the peak of its field at the pixel-centre position `center * L - 0.5`
  on each axis. It used to round `center * L` to a whole pixel (half to even), and on an even-length axis the
  Gaussian kernel took another half pixel off, so the peak moved with the parity of the axis: `center=0.5` sat
  one pixel past the middle of a 3-, 7- or 11-pixel axis but on it for a 5- or 9-pixel one. The drawn field now
  shifts by up to one pixel, so outputs under a fixed seed change. Fixes #4811. ([#4990](https://github.com/kornia/kornia/pull/4990))
- `kornia.geometry.homography` now defines `__all__` with its eight public functions, so its imports are no longer
  star-imported into `kornia.geometry`: `torch`, `warnings`, `typing.Optional`, `typing.Tuple`, the type alias
  `TupleTensor`, `KORNIA_CHECK_SHAPE`, `safe_inverse_with_mask` and `safe_solve_with_mask`. None of them is documented
  on `kornia.geometry`, and per `docs/source/get-started/stability.rst` undocumented helpers are private. They are
  dropped from `tests/api_surface.json` in this change. (#4994) ([#4994](https://github.com/kornia/kornia/pull/4994))
- `decompose_essential_matrix` and `motion_from_essential` keep the leading dims of the input for an unbatched
  `(3, 3)` essential matrix: the rotations are now `(3, 3)` and the four candidates `(4, 3, 3)`, matching `t` of shape
  `(3, 1)` and `(4, 3, 1)`. They used to come back as `(1, 3, 3)` and `(1, 4, 3, 3)`, so a caller that indexed `R1[0]`
  after an unbatched call has to drop the index. Batched inputs are unchanged. Fixes #4878. ([#4998](https://github.com/kornia/kornia/pull/4998))
- `RANSAC` defaults changed: `score_type` is now `"msac"` (was `"ransac"`) and `lo_sample_size` is now `32` (was `None`, iterative full-inlier refits). Candidates are now ranked by truncated squared residuals instead of inlier count, and local optimization fits `max_lo_iters` random 32-inlier subsets before one full-inlier refit whenever the consensus set has more than 32 inliers. Estimates for a given seed change accordingly; pass `score_type="ransac"` and `lo_sample_size=None` to keep the previous behavior. The RANSAC time-mAA benchmark now compares the class defaults, with and without PROSAC, against PoseLib and OpenCV, and skips PROSAC configurations on pairs without a match ranking. ([#5002](https://github.com/kornia/kornia/pull/5002))
- `So2` reads a `(B, 1)` complex number or angle as `(B,)`: an element built by `So2(z)` or `So2.exp(theta)` from a column
  now has `z` and `log()` of shape `(B,)` and `matrix()` of shape `(B, 2, 2)`, `So2.hat(theta)` returns `(B, 2, 2)`, and
  `So2 * points` rotates `(B, 2)` points pairwise into `(B, 2)`. They used to keep the column, so `z` and `log()` were
  `(B, 1)`, `matrix()` and `hat()` were `(B, 1, 2, 2)` (which `vee()` rejects), `So2 * points` broadcast into `(B, B, 2)`
  with every rotation applied to every point, and `Se2` rejected the element. The given tensor is still what the module
  registers, so a `(B, 1)` `nn.Parameter` stays its parameter and `state_dict()` keeps the `(B, 1)` shape. `(B,)` and `()`
  inputs are unchanged. Fixes #4932. ([#5005](https://github.com/kornia/kornia/pull/5005))
- `kornia.filters.median_blur`, `MedianBlur` and `RandomMedianBlur` now reject an even or non-positive `kernel_size` entry with kornia's `BaseError` ("Kernel size must be an odd integer bigger than 0"), and `RandomMedianBlur` does so at construction. Previously `RandomMedianBlur` accepted such a kernel at construction, and the call failed with a raw torch `RuntimeError` from a reshape, a pad or a convolution, which `except RuntimeError` no longer catches. The one input that used to pass was an even kernel on an empty batch or an image with zero channels, which returned an empty tensor and now raises too. Integral floats and numpy integers such as `(3.0, 3.0)` are still accepted. ([#5007](https://github.com/kornia/kornia/pull/5007))
- `Se2.t` / `Se3.t` (and `.translation`) are now always a tensor registered as module state, whichever constructor built
  the pose: `__init__` unwraps a `Vector2`/`Vector3`, and `identity`, `random` and `Se3.from_qxyz` pass tensors. Those
  three used to keep a `Vector2`/`Vector3` outside module state, so `Se3.identity(1).t[..., 0]` raised `RuntimeError:
  Could not infer dtype of ellipsis`, `Se2.identity(1) * points` returned a `Vector2`, and `state_dict()` and `.to()`
  skipped the translation. Their translation is now an `nn.Parameter` with `requires_grad=True` like the other
  constructors', so `.x`/`.y`/`.z` on it stop working (index the tensor instead) and `matrix()`, `adjoint()` and `g * p`
  of such a pose carry a `grad_fn`; `g * p` still returns a `Vector2`/`Vector3` when `p` is one. Numerical results are
  unchanged. A `state_dict` saved before this change has no `_translation` entry for such a pose, so a strict
  `load_state_dict` of it now raises `Missing key(s)`; load it with `strict=False`, which keeps the translation the
  constructor built, as loading did before. (#5011) ([#5011](https://github.com/kornia/kornia/pull/5011))
- `fit_line` with `weights` now centres `D >= 3` points on the weighted centroid and returns it as the origin, as the
  `D = 2` branch does, so a point with weight 0 no longer moves the origin or tilts the direction. Previously the points
  were centred on the unweighted mean, which was also the returned origin, and only the scatter matrix was weighted, so
  any non-uniform weights gave a different origin and direction from weighted least squares. Uniform weights give the
  same line as before, and a row whose weights sum to 0 still gets the unweighted mean as its origin. ([#5029](https://github.com/kornia/kornia/pull/5029))
- `RANSAC` for homographies and fundamental matrices now defaults to `local_optimization="lm"`: it keeps the eight best minimal models of the run, refines them with Levenberg-Marquardt on the truncated Sampson or transfer error after sampling, and refines the winner on its inliers with a Cauchy loss. Previously each new best model was refitted with DLT solutions on 32-inlier subsets and then on all its inliers (`lo_sample_size=32`). `model_type="fundamental"` now draws seven-point samples like `"fundamental_7pt"`; it used to draw eight-point ones, which the new `"fundamental_8pt"` keeps. With the new default, auto batches grow from 256 samples on CPU instead of a fixed size, early stopping follows the support of the best minimal model instead of the refitted one, and the returned mask holds the inliers of the returned, refined model. Estimates for a given seed change accordingly; pass `local_optimization="dlt"` (with `"fundamental_8pt"` for fundamental matrices) to keep the previous behavior. Essential matrices and line-segment homographies are unchanged. ([#5031](https://github.com/kornia/kornia/pull/5031))
- `homography_warp` with `normalized_homography=False` now passes `mode` and `align_corners` on to `warp_perspective`. Previously that path always sampled with `mode="bilinear"` and `align_corners=True`, whatever the caller passed. A call with the default arguments now uses the signature's `align_corners=False`: with `padding_mode="zeros"` or `"border"` the output moves only by float rounding, but with `padding_mode="reflection"` the out-of-bounds samples change. Pass `align_corners=True` to get the previous output. Callers who passed `mode="nearest"` or `mode="bicubic"` now get that interpolation instead of bilinear. ([#5032](https://github.com/kornia/kornia/pull/5032))
- `save_pointcloud_ply` and `save_pointcloud_ply_binary` now write every row of the flattened input, including rows whose x, y and z are all non-finite (written as `nan` / `inf` in ASCII and as IEEE values in binary), so the `element vertex` count equals the number of points and row `i` of the file is row `i` of the input. Previously such rows were silently skipped and the vertex count excluded them, which broke the row-to-pixel correspondence of an organised cloud such as `depth_to_3d` output with invalid depth. ([#5033](https://github.com/kornia/kornia/pull/5033))
- `spatial_softmax2d`, `spatial_soft_argmax2d` and `SpatialSoftArgmax2d` now treat `temperature` as a softmax
  temperature, as `conv_soft_argmax2d` and `conv_soft_argmax3d` already did: the input is divided by it, so a smaller
  value gives a sharper distribution. Previously the input was multiplied by `temperature`, so a larger value was
  sharper; a Python float raised `AttributeError`; and `0` or a negative value was accepted (`0` returned a uniform map,
  a negative value inverted it). A float is now accepted, and a temperature that is not positive raises `ValueError`.
  To keep an old result, pass `1 / temperature`. The default, `1.0`, gives the same output as before. ([#5035](https://github.com/kornia/kornia/pull/5035))
- `render_gaussian2d` now normalises each axis with a softmax over the Gaussian's exponents instead of dividing by the
  sum plus `1e-8`. A Gaussian inside the image now sums to 1 up to roundoff; the `1e-8` term used to leave it about
  `1e-8` short. A mean far outside the grid now renders the grid part of the Gaussian rescaled to sum to one, which
  puts its mass on the nearest border row or column, in every dtype and with finite gradients. Previously the `1e-8`
  term made such a heatmap lose its mass once the mean was about six standard deviations outside the grid, and it was
  exactly zero once every sample underflowed. ([#5036](https://github.com/kornia/kornia/pull/5036))
- `ParametrizedLine.intersect` now returns two `torch.Tensor`s, as annotated. Previously the lambda and the point came
  back as `kornia.geometry.vector.Scalar` wrappers, because the plane's offset is a `Scalar`, and
  `Hyperplane.signed_distance` rejected the returned point. Code that unwrapped them with `.data` still gets the same
  values, but `.data` on a tensor is detached from autograd: drop the `.data` to keep gradients. ([#5044](https://github.com/kornia/kornia/pull/5044))
- `strict_maxima_bonus` is deprecated and ignored in `conv_soft_argmax3d`, `ConvSoftArgmax3d`, `conv_quad_interp3d`, `ConvQuadInterp3d`, `iterative_quad_interp3d`, `IterativeQuadInterp3d` and `AdaptiveQuadInterp3d`. Previously the quad-interp functions and modules added `strict_maxima_bonus=10.0` by default to the returned value at every converged NMS maximum, and `conv_soft_argmax3d` scaled its values when given a positive bonus, incorrectly (#5018). Now no bonus is added: the returned values are the refined responses. The argument is still accepted, by keyword or in its 0.8.3 position, but it emits a `DeprecationWarning` and will be removed in a future release. The default refiners of `ScaleSpaceDetector` and `SIFTFeatureScaleSpace` already passed `strict_maxima_bonus=0.0`, so their detections are unchanged. Code that ranked the dense value map returned by these functions with `topk` and relied on the bonus to put strict maxima first should select candidates with a `nms3d(x, (3, 3, 3), mask_only=True)` mask instead. ([#5045](https://github.com/kornia/kornia/pull/5045))
- `ParametrizedLine.through`, `fit_line`, `fit_plane` and `Hyperplane.through` now raise `ValueCheckError` for input that does not determine a line or a plane: two coincident points for `ParametrizedLine.through`; fewer than two points, identical points, or weights that do not sum to a positive value for `fit_line`; fewer than three points, identical points or collinear points for `fit_plane` and `Hyperplane.through`. They used to return an arbitrary valid-looking result: a zero direction, a fixed direction such as (0, 1) or (1, 0, 0), a NaN origin for all-zero weights, or one of the infinitely many planes that contain the line. A batch is rejected when any of its rows is degenerate. The collinearity tolerance is relative and scales with the machine epsilon of the input dtype. The checks are skipped under `torch.compile` and after `disable_checks()`, where the previous results come back. ([#5054](https://github.com/kornia/kornia/pull/5054))
- `Hyperplane.through` now divides the cross product by its Euclidean norm, so the returned normal has unit length and `signed_distance`, `abs_distance` and `projection` are right for tilted planes. It used to divide by the p = -1 "norm", so the normal of a plane through three generic points was not unit: for the plane through (1, 0, 0), (0, 2, 0) and (0, 0, 3) it was (-6, -3, -2), and distances came out 7 times too large. Axis-aligned planes, whose cross product has a zero component, used to take the SVD fallback, whose sign is arbitrary; they now take the cross-product branch, so their normal is `(p2 - p0) x (p1 - p0)` normalised and can be the negation of the previous one, which flips the sign of `signed_distance`: the plane through (0, 0, 1), (1, 0, 1) and (0, 1, 1) now has the normal (0, 0, -1) instead of (0, 0, 1). The collinearity threshold is now evaluated per batch row, relative to the dtype's machine epsilon, instead of once over the whole batch. ([#5055](https://github.com/kornia/kornia/pull/5055))
- `conv_soft_argmax2d`, `conv_soft_argmax3d`, `ConvSoftArgmax2d` and `ConvSoftArgmax3d` now measure each window's
  softmax-weighted offset from the window centre in pixels, so a window returns the softmax expectation of its pixel
  coordinates, as the docstring formula states. Previously the offset kernel ran from −1 to 1 along every axis whatever
  the window size, which scaled the offset by 2/(k − 1): a 5-wide window reported a peak at x = 8 as x = 7, a 7-wide
  window as x = 6, and a 2-wide window doubled the offset. Window sizes 1 and 3 (the default) give the same output as
  before; any other `kernel_size` entry changes the coordinates along that axis, in pixels and in normalized coordinates.
  (#5017) ([#5059](https://github.com/kornia/kornia/pull/5059))
- `conv_quad_interp3d`, `iterative_quad_interp3d`, `ConvQuadInterp3d`, `IterativeQuadInterp3d` and `AdaptiveQuadInterp3d`
  now reject a maximum whose last iteration still moves its integer centre: it keeps its grid coordinates and its input
  value. Previously such a point was reported at the new centre plus the shift solved at the old one, one voxel beyond
  that fit on every axis that moved, with its value read at the moved voxel, and a point alternating between two cells
  landed one voxel off on either side depending on the parity of `n_iters`. A rejected maximum, whatever rejected it, now
  reads its value at the voxel it reports instead of the voxel it last moved to. With the default `n_iters=5` this
  changes a few percent of the refined maxima (1.4% to 8.8% on blurred-noise inputs), so `ScaleSpaceDetector` and the
  detectors built on it, such as `SIFTFeatureScaleSpace` with its default `descriptor_backend="patch"`, return different
  keypoint positions, scales and responses. (#5038) ([#5060](https://github.com/kornia/kornia/pull/5060))
- `load_image` with `ImageLoadType.RGBA8` now fills the alpha channel it adds to RGB and grayscale images with 255, so
  they load opaque. It used to fill that channel with 0, which made them fully transparent. Images that already have an
  alpha channel keep their own. ([#5077](https://github.com/kornia/kornia/pull/5077))
- `RANSAC("essential")` now defaults to `local_optimization="lm"`: the run's eight best five-point models are refined by Levenberg-Marquardt on the essential manifold, and the winner by a final Cauchy refinement on its inliers, in the caller's calibrated coordinates. Previously `"essential"` only supported `"dlt"`, which refits eight-point fundamental matrices on inlier subsets and projects them onto the manifold; `local_optimization="dlt"` still selects that loop, but its five-point samples now come from the new `find_essential` solver, so its estimates can differ from before at rounding level and in sign. Estimates for a given seed change; the returned E has unit Frobenius norm and a positive largest entry, and the returned mask holds the inliers of the returned matrix. The returned E no longer carries autograd; its gradient to the correspondences was NaN before. Automatic batches for essential matrices now start at 64 samples on CPU and double up to 1024, and start at 256 on CUDA and MPS and double up to 8192, instead of a fixed 128 to 512 samples per batch on CPU and 2048 on CUDA and MPS. ([#5078](https://github.com/kornia/kornia/pull/5078))
- `mean_average_precision` now computes each class's recall over the ground-truth objects of that class, as the
  reference implementation it is adapted from does. It used to divide by the number of ground-truth objects of every
  class, so when the ground truth held more than one label no class could reach recall 1 and the per-class AP and the
  mAP came out too low: two objects of different classes, each detected exactly, scored 0.5455 instead of 1. Inputs
  whose ground-truth objects all share one label give the same values as before. ([#5079](https://github.com/kornia/kornia/pull/5079))
- `kornia.morphology.dilation`, `erosion`, `opening`, `closing`, `gradient`, `top_hat` and `bottom_hat` now raise a `TypeError` for an image that is not floating point (`bool`, `uint8` or another integer dtype), before any padding or engine dispatch. Such images were never supported but used to run, with results that depended on the dtype, the border and the device: under the default `border_type="geodesic"`, `dilation` of a `bool` image on CPU set every border pixel, a `uint8` image raised torch's overflow `RuntimeError` on CPU and returned `240` at the border on MPS, and an integer image kept the finite `max_val` sentinel for excluded kernel cells, so `dilation` of `[0, 50000, 0]` by `[[1, 0, 1]]` returned `[50000, 40000, 50000]` instead of `[50000, 0, 50000]`. Floating-point images are unaffected. Convert an integer or boolean image with `.float()` first. ([#5081](https://github.com/kornia/kornia/pull/5081))
- `conv_soft_argmax2d`, `conv_soft_argmax3d`, `ConvSoftArgmax2d` and `ConvSoftArgmax3d` now continue the window-centre
  coordinate grid into the padding, so a window whose centre straddles the border reports the softmax expectation of its
  in-image pixels. Previously the grid was zero-padded: with an even `kernel_size` and `padding = k / 2` (for example
  `k = 2` with the default `padding = 1`), the first and last window on an axis averaged the coordinate 0 of a padded
  pixel into its centre on every channel and reported a point far inside the image (x = 3.5, y = 1.25 for a peak at
  x = 8, y = 2 on a 5 x 9 map). Odd windows, and even windows with `padding < k / 2`, give the same output as before.
  (#5066) ([#5086](https://github.com/kornia/kornia/pull/5086))
- `conv_quad_interp3d`, `iterative_quad_interp3d` and their modules now treat a fit as singular when `|det H| <= 1e-7 * m**3`, with `m` the largest magnitude among the Hessian entries. Previously the floor was an absolute `|det H| <= 1e-7`, and the determinant is cubic in the response amplitude, so well-conditioned maxima of a low-amplitude response kept their integer coordinates: `ScaleSpaceDetector` on a `[0, 1]` image returned almost every keypoint unrefined. Those keypoints are now refined to sub-pixel positions, which changes detector output. The solve is scaled so that its largest Hessian entry is about 1, so a flat image region, whose Hessian entries are rounding noise, keeps a finite `float32` image gradient. (#5065) ([#5087](https://github.com/kornia/kornia/pull/5087))
- `fit_line` now raises `ValueCheckError` when the points with positive weight are all the same point, for example when
  the weights are positive on a single point: a point with weight 0 does not count, so such a row does not determine a
  line. It used to return the fallback direction (0, 1) in 2-D and (1, 0, 0) in 3-D. A batch is rejected when any of
  its rows is degenerate. The check is skipped under `torch.compile` and after `disable_checks()`, where such a row
  still gets (1, 0, 0) in 3-D, and (1, 0) in 2-D with the total least squares fit of #5094. (#5082) ([#5088](https://github.com/kornia/kornia/pull/5088))
- `fit_line` now fits 2-D points by total least squares, as it does points of higher dimension: the line minimises the
  perpendicular distances to the points. It used to fit an ordinary least squares slope of y on x, which minimises the
  vertical distances, so noisy 2-D points gave a direction biased towards horizontal (8.3 degrees off for a near-vertical
  set whose total least squares fit is 1.4 degrees off), swapping x and y changed the line, and a set with
  `sum((x - x.mean()) ** 2) <= 1e-8`, such as a slanted set in small units, came out vertical. In `float16`, steep sets
  and sets spanning a few hundred units came out NaN or `(1, 0)`. Points exactly on a line give the same direction as
  before: the 2-D direction still has a non-negative x component, and an exactly vertical line still gets `(0, 1)`.
  Under `torch.compile` or `disable_checks()`, where degenerate input is not rejected, identical 2-D points now get the
  direction `(1, 0)` instead of `(0, 1)`. ([#5094](https://github.com/kornia/kornia/pull/5094))
- `ImageRegistrator`'s `'similarity'` and `'rotation'` models now rotate the image in pixels on a non-square image
  (#5063). `Similarity.forward()` used to apply `scale * angle_to_rotation_matrix(rot)` directly in the normalized
  `[-1, 1]` frame, whose x and y units differ by `W / H`, so on an `H != W` image the model was not a rotation of the
  pixels and a rotated image could not be registered exactly: a 6 degree rotation of a 32 x 48 image was recovered with
  a 1.2 px corner error, against 0.001 px on a 40 x 40 image. `register()` now calls the new
  `Similarity.set_image_shape(height, width)` with the size of `dst_img`, and `forward()` then uses
  `scale * diag(H / W, 1) @ angle_to_rotation_matrix(rot) @ diag(W / H, 1)`, so `rot` and `scale` are the rotation and
  scaling of the pixels about the image centre. On a non-square image the matrices returned by `register()` and used by
  `warp_src_into_dst()` and `warp_dst_into_src()` change. Square images, the `'translation'` and `'scale'` models, and a
  `Similarity` whose image shape was never set are unchanged. ([#5096](https://github.com/kornia/kornia/pull/5096))
- `fundamental_from_projections` now divides a float16 result by its largest absolute entry before casting it back
  to float16, so pixel-unit projection matrices no longer give `inf` (#4877). It used to cast the unnormalised
  float32 determinants back, which overflowed for ordinary cameras (5 of 9 entries `inf` for focal lengths of
  700-800 px) and flushed projection matrices with entries of order `1e-2` to all zeros. float16 results that used
  to be finite now differ from before by a positive scale factor, up to rounding; an all-zero result stays zero.
  float32, float64 and bfloat16 results are unchanged. ([#5097](https://github.com/kornia/kornia/pull/5097))
- `mean_average_precision` now compares float64 recalls against the recall thresholds `i / 10`, so a recall that lands
  exactly on a tenth reaches it. Previously the thresholds were the float32 values of `torch.arange(0, 1.1, 0.1)` widened
  to Python floats, and `0.10000000149`, `0.20000000298`, `0.30000001192`, `0.40000000596`, `0.60000002384` and
  `0.80000001192` lie above the tenths they stand for. A float64 recall of exactly 0.1, 0.2, 0.3, 0.4, 0.6 or 0.8 then
  skipped that threshold, and the class AP and the mAP came out lower than in float32: 10 objects with detections ranked
  TP, FP, then 9 TP scored 0.9174 (111 / 121) instead of 0.9256 (112 / 121). float64 now gives the float32 values.
  float16, bfloat16 and float32 results are unchanged, because each threshold rounds to the same value in those dtypes
  (#5083). ([#5101](https://github.com/kornia/kornia/pull/5101))
- `Hyperplane.through` now takes the edges and the cross product of float16 and bfloat16 points in float32 and casts the
  unit normal back, so the normal keeps the `(p2 - p0) x (p1 - p0)` orientation. Previously, in float16, the cross
  product of a small valid triangle underflowed (edges of 1e-4 give 1e-8, below the smallest subnormal) and that of a
  large one overflowed (edges of 300 give 9e4, above 65504), and the triangle took the SVD fallback, whose sign is
  arbitrary: the right triangle at the origin with legs of 1e-4 or 300 came back as `(0, 0, 1)` instead of
  `(0, 0, -1)`, which flips `signed_distance` for every point. Triangles with edges near the float16 limit were rejected
  as collinear, and an edge longer than 65504 gave nan. Without the value checks, as under `torch.compile`, a float16 or
  bfloat16 triangle now takes the SVD fallback at the float32 threshold, so a thin one keeps its orientation there too.
  The collinearity check keeps the tolerance of the input dtype. float32 and float64 results are unchanged (#5064). ([#5104](https://github.com/kornia/kornia/pull/5104))
- `fit_line` now raises `ValueCheckError` when a weight is negative. Such weights passed the check that each row's
  weights sum to a positive value, but a negative weight can make the weighted scatter matrix indefinite, and then the
  2-D and the D ≥ 3 fits of the same points disagree: for the four 2-D points of #5106 with weights `(1, -3, 1, 2)`, the
  3-D fit of the points with z = 0 returned the direction (0.79, 0.62, 0), while the 2-D fit returned the vertical
  fallback (0, 1), or, with the total least squares fit of #5094, the perpendicular direction (0.62, -0.79). Any
  negative weight is rejected, also one small enough to leave the scatter without a negative eigenvalue, such as -0.01
  among weights of 1. A weight of 0 or -0.0 is still accepted and drops its point. A batch is rejected when any of its rows
  has a negative weight. The check is skipped under `torch.compile` and after `disable_checks()`, where negative
  weights are still fitted. (#5106) ([#5108](https://github.com/kornia/kornia/pull/5108))
- `RgbToGrayscale()` without `rgb_weights` now stores `rgb_weights = None` instead of the float32 tensor
  `[0.299, 0.587, 0.114]`, and raises `TypeError` for an integer or bool image other than uint8, as `rgb_to_grayscale`
  does. It used to return all zeros for int8, int16, int32 and int64 images. ([#5111](https://github.com/kornia/kornia/pull/5111))
- `RandomCrop` and `RandomCrop3D` previously exposed `transform_matrix` in padded-canvas coordinates, so callers had to add the pre-crop left/top (and front for 3D) padding themselves. They now map original input coordinates directly to output coordinates; callers that manually added padding must remove that workaround. ([#5114](https://github.com/kornia/kornia/pull/5114))
- `sampson_epipolar_distance`, `symmetrical_epipolar_distance`, `left_to_right_epipolar_distance`,
  `right_to_left_epipolar_distance`, `point_line_distance`, `oneway_transfer_error` and `symmetric_transfer_error` no
  longer add `eps` to their denominators or inside the square root, and return `inf` in the singular cases instead
  (#4881): both epilines with `a = b = 0` for the Sampson distance, either one for the symmetrical distance, the line
  for `point_line_distance` and the one-way epipolar distances, and a point that `H` maps to `w' = 0` for the transfer
  errors. The values are now independent of the scale of `F`, `H` or the line, `squared=False` returns exactly 0 for an
  exact match, with a zero gradient, and the CUDA matmul and the CPU manual Sampson paths agree. Before, the same `F` at
  `‖F‖ = 1e-3` scored 90% lower than at `‖F‖ = 1` (9.3e-6 relative there), an exact match scored `sqrt(eps) = 1e-4`,
  the two Sampson paths differed by that scale-dependent amount, a degenerate `F` scored `residual² / eps` (`1e8` for
  `F = diag(0, 0, 1)`) on CPU and `inf` on CUDA, and on CPU an all-zero `F` or line put every point at distance 0. The
  `eps` parameters are still accepted and have no effect. ([#5116](https://github.com/kornia/kornia/pull/5116))
- `MS_SSIMLoss` now pairs every channel with every scale and averages the MS-SSIM term over the channels. Previously
  its masks were laid out scale-major while the grouped convolution reads them channel-major, so for RGB input red was
  filtered at scales 0, 0, 0, 1, 1, green at 1, 2, 2, 2, 3 and blue at 3, 3, 4, 4, 4, and the luminance term saw only
  blue: with `alpha=1` and `compensation=1`, at the centre of a random 32 x 32 image, brightening the red or green
  channel by 0.3 cost less than 1e-4 while brightening blue cost 0.28, and swapping RGB to BGR changed the loss. The
  contrast-structure terms of all channels were multiplied together, single-channel input got the MS-SSIM term cubed,
  and inputs whose channel count does not divide `3 * len(sigmas)` (2 or 4 channels with the default sigmas) raised
  `RuntimeError`. The MS-SSIM term is now one minus the mean, over the channels, of the per-channel MS-SSIM (luminance
  at the coarsest scale, contrast-structure at every scale), as in the reference implementation of Zhao et al., and any
  channel count is accepted. With `alpha > 0` the loss changes for differing images; the Gaussian-weighted L1 term is
  unchanged. The masks are rebuilt from `sigmas` instead of being persisted: old checkpoints still load strictly, but
  their `_g_masks` values are ignored. ([#5121](https://github.com/kornia/kornia/pull/5121))
- `conv_soft_argmax2d`, `conv_soft_argmax3d`, `ConvSoftArgmax2d` and `ConvSoftArgmax3d` now take the softmax of every
  window over that window's own values (#5020), so a window's coordinates and value no longer depend on the rest of the
  map. Previously the exponent was shifted by the maximum of the whole map and `eps` was added to each window's
  denominator: a window 15 or more units of `temperature` below that maximum was pulled toward its centre and its value
  toward 0, and from about 30 units it returned its centre with value 0 in float32, float64 and bfloat16; in float16 it
  was wrong from about 15 units and NaN from about 18. With `eps=0` such a window was right until `exp` underflowed and
  NaN beyond (about 95 units in float32 and bfloat16, 745 in float64). Windows near the map maximum move within
  precision: on maps spanning a few units of `temperature`, by up to about 3e-6 in float32 and float64 and by one to
  three ulp in float16 and bfloat16. `eps` is still accepted and has no effect. `conv_soft_argmax3d` now runs in float16
  and bfloat16 on CPU, where it raised `NotImplementedError` (`avg_pool3d`). A `padding` above half of `kernel_size`, or
  a negative one, raises `ValueError` instead of the pooling kernel's `RuntimeError`. The windows are unfolded, which
  takes `k*k` (2D) or `k*k*k` (3D) times the memory of the output map, and a CPU forward pass takes about 1.2 to 2.4
  times as long as before (3x3 to 15x15 windows). ([#5134](https://github.com/kornia/kornia/pull/5134))
- `line_segment_transfer_error_one_way` returns the mean perpendicular distance, in pixels, of the two mapped image-1
  endpoints from the image-2 line. It used to return that distance multiplied by the image-2 segment length, because the
  line was not normalised: the same 3-px offset scored 197 on a 66-px segment and 994 on a 331-px one. Values drop by the
  image-2 segment length, so a threshold tuned on the old values has to be re-tuned in pixels.
  `RANSAC("homography_from_linesegments")` and `find_homography_lines_dlt_iterated` already scored the pixel distance
  and are unchanged up to rounding. Fixes #4867. ([#5135](https://github.com/kornia/kornia/pull/5135))
- `MS_SSIMLoss` now builds an odd, centred Gaussian window of `2 * int(2 * sigmas[-1]) + 1` pixels (#5126). Previously
  the window was `int(4 * sigmas[-1] + 1)` wide, which is even for a largest sigma such as 1.3 (6 pixels) or 3.3 (14):
  the window sat half a pixel off centre and the `reduction="none"` map came out `(N, H - 1, W - 1)` instead of the
  documented `(N, H, W)`. Those sigmas now get a window one pixel narrower (5 and 13), the map keeps the input shape and
  the loss changes by about 2% on random images. Every integer or half-integer sigma, the defaults included, keeps its
  window size, and its outputs are bitwise unchanged. ([#5143](https://github.com/kornia/kornia/pull/5143))
- `lovasz_softmax_loss` and `LovaszSoftmaxLoss` now compute each class's Lovász gradient from that class's binary
  foreground mask, as the reference implementation of Berman et al. does (`lovasz_softmax(..., classes="all",
  per_image=True)`). Previously they used the integer class labels in place of the mask, so the loss and its gradients
  did not match the reference and changed when the classes were renumbered: for the three-pixel, three-class example
  in #5150 the loss was 0.2167, or 0.2389 after relabelling the classes, instead of 1/3. Loss values and gradients
  change. ([#5151](https://github.com/kornia/kornia/pull/5151))
- `Resize`, `LongestMaxSize`, `SmallestMaxSize` and slice-mode `RandomResizedCrop` with `align_corners=False` and bilinear or bicubic resampling now record the half-pixel map their resize samples with, `x' = (x - x0 + 0.5) * W_out / W_crop - 0.5` and likewise for `y`, as `transform_matrix`, where `x0` and `W_crop` are the crop's first column and width (`0` and the input width for the resize classes). They used to record the corner-to-corner map `x' = (x - x0) * (W_out - 1) / (W_crop - 1)`, so keypoints and boxes moved through `AugmentationSequential` landed up to `|W_out / W_crop - 1| / 2` output pixels from the content they annotate (half a pixel at a 2x upscale), and the `inverse()` of the three resize classes warped the image back with that map, which did not return content to its original place. The resized images, `align_corners=True` (the default for these classes), nearest resampling and resample-mode crops are unchanged. ([#5179](https://github.com/kornia/kornia/pull/5179))
- `blur_pool2d`, `BlurPool2D`, `max_blur_pool2d` and `MaxBlurPool2D` with an even kernel size now return
  `ceil(n / stride)` samples along each axis, as odd kernel sizes already did, where `n` is the input height or width,
  reduced by `max_pool_size - 1` for the max variants. The input is zero-padded by `(k - 1) // 2` pixels before and
  `k // 2` after, the amounts antialiased-cnns uses (antialiased-cnns pads by reflection by default; kornia pads with
  zeros). Previously both sides were padded by `(k - 1) // 2`, so an axis lost its last sample whenever `n - 1` was a
  multiple of the stride. At stride 1 that is every axis: a 7x10 map returned 6x9 and now returns 7x10. At stride 2 only
  an odd `n` is affected: `blur_pool2d` on a 7x10 map returned 3x5 and now returns 4x5 while 8x8 is unchanged, and
  `max_blur_pool2d` with the default `max_pool_size=2` on an 8x8 map returned 3x3 and now returns 4x4. The old output is
  always the leading block of the new one. Odd kernel sizes are unchanged, bit for bit (#5166). ([#5184](https://github.com/kornia/kornia/pull/5184))
- `kornia.core.load_state_dict_from_url` now loads checkpoints with `weights_only=True` unless the caller passes
  `weights_only=False`, and so does every `pretrained=True` model built on it. The loaders that call `torch.load`
  directly pass `weights_only=True` as well: `ModelBase.load_checkpoint` for a local file, `SmallSRNet`,
  `RRDBNetBuilder` and LightGlue's local weights. Previously the wrapper forwarded the `weights_only=False` default of
  `torch.hub.load_state_dict_from_url` on every supported torch, and the direct calls used `torch.load`'s default, which
  is `False` before torch 2.6, so loading a checkpoint unpickled whatever objects it stored. Now only tensors, primitive
  types and plain containers load. A checkpoint holding any other type fails with `pickle.UnpicklingError`, which
  `load_state_dict_from_url` chains to its `RuntimeError`. To load such a file through any of these loaders, allowlist
  its types with `torch.serialization.safe_globals([...])` around the call. `load_state_dict_from_url` also accepts
  `weights_only=False`, which the other loaders do not; pass it only for a file you trust. All of kornia's pretrained
  checkpoints load under the new default, on torch 2.14 and 2.5.1, without an allowlist. On torch before 2.10,
  `weights_only=True` does not guarantee that a checkpoint runs no code: PyTorch's advisories
  [GHSA-53q9-r3pm-6pq6](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6) (before 2.6) and
  [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p) (before 2.10)
  report checkpoints crafted to run code despite it. Load a checkpoint from a source you do not trust only on torch 2.10
  or later.

  The fallback download sources on `cmp.felk.cvut.cz` (HardNet8, LoFTR, SOLD2, DexiNed and three LightGlue heads) are
  now fetched over `https` instead of `http`. DeFMO's `http` fallback on `ptak.felk.cvut.cz` has no usable `https`
  equivalent and is removed, so DeFMO loads from its Hugging Face mirror only. ([#5225](https://github.com/kornia/kornia/pull/5225))
- `KORNIA_CHECK_IS_IMAGE` checks the shape again with the default `raises=True`, as its docstring says and as it did
  before #3156. It used to return `True` for a tensor that is not `(*, 1, H, W)` or `(*, 3, H, W)`, such as `(5, 4, 5)`,
  an RGBA `(B, 4, H, W)`, `(4, 5)` or a 0-d scalar, while the same call with `raises=False` returned `False`. It now raises
  `ImageError`, so code that passed a non-image through the check gets `ImageError`. So does `image_to_string`, which
  rendered a 6-channel tensor as if it had 3 channels. A float image containing NaN used to pass the range check for
  both `raises` values. It now fails it: `raises=True` raises `ValueCheckError` and `raises=False` returns `False`. ([#5232](https://github.com/kornia/kornia/pull/5232))
- kornia's validation errors now also derive from the matching built-in exception: `ShapeError`,
  `ValueCheckError`, `DeviceError` and `ImageError` are `ValueError`s, and `TypeCheckError` is a
  `TypeError`. Previously all five derived only from `BaseError` (an `Exception`), so an
  `except ValueError:` or `except TypeError:` around a kornia call caught the functions that raise the
  built-ins themselves but let a failed `KORNIA_CHECK_*` escape, for example `gaussian_blur2d` on a
  2-D tensor. Such clauses now catch these errors as well; `except BaseError` and the specific classes
  keep working. A failed plain `KORNIA_CHECK` still raises the bare `BaseError`, which is neither
  built-in. This does not restore v0.8.2, where the shape, device and color/gray image checks raised
  `TypeError`: they are `ValueError`s now, so an `except TypeError` written for v0.8.2 still misses
  them. Separately, a call whose tensors share a device but differ in dtype, such as
  `find_homography_dlt` with a `float32` and a `float64` point set, `get_motion_kernel2d`,
  `geometry.transform.Affine` or an augmentation's parameter generator given mixed dtypes, now raises
  `TypeCheckError` ("expected torch.float32, got torch.float64") instead of a `DeviceError` that
  listed the same device twice; `except DeviceError` no longer catches it. When the tensors span more
  than one device, the call raises `DeviceError` as before, whether or not their dtypes also differ
  and wherever the mismatches sit in the argument list; its message now names only the two devices
  ("Passed tensors are not on the same device: expected cpu, got cuda:0."). ([#5233](https://github.com/kornia/kornia/pull/5233))
- `ImageModule`, `ImageSequential` and the augmentation containers now convert NumPy, PIL and
  image-path inputs by their dtype, and return channels-last arrays for `output_type="numpy"`.
  `to_tensor` used to divide every input by 255 whatever its dtype: a float image already in `[0, 1]`
  shrank 255 times (0.5 became 0.00196), a `uint16` image or 16-bit PNG reached 257, and a `bool` mask
  became 1/255. Now an integer input is divided by the maximum of its dtype (`uint8` by 255 as before,
  `uint16` by 65535, `int16` by 32767), a `bool` array becomes 0 and 1, and a floating array keeps its
  values; `uint8` inputs convert to the same values as before, and a PIL input now takes the default
  dtype, as a NumPy input already did (it was always `float32`). A signed integer image maps to
  `[iinfo.min / iinfo.max, 1]`, so its negative values stay negative (`int8` -128 gives -128 / 127).
  `output_type="numpy"` (and `to_numpy`) used to return the `(C, H, W)` / `(B, C, H, W)` layout of the
  output tensor, which `to_tensor` read back as channels-last and transposed; it now returns `(H, W,
  C)` / `(B, H, W, C)`, the layout NumPy inputs use, with the tensor's values, so a float output fed
  back in converts to the same tensor. Every 3-D or 4-D output is treated as an image, so a non-image
  element of a tuple output (such as a `(B, N, 4)` box tensor) moves its axis too; modules with
  several outputs are #5210. Code that expected the old layout moves the channel axis with
  `np.moveaxis(out, -1, -3)`. ([#5239](https://github.com/kornia/kornia/pull/5239))
- A missing optional dependency now raises by default. `kornia_config.lazyloader.installation_mode` used to default to
  `"ask"`, which asked on stdin whether to install the package: without an interactive stdin (a CI job, a DataLoader
  worker) that raised `EOFError` instead of `ImportError`, and on an open stdin that nobody writes to it blocked. The
  default is now `"raise"`: an `ImportError` naming `pip install "kornia[<extra>]"`. `"ask"` is opt-in and raises the same
  `ImportError` without an interactive terminal on stdin and stdout (for example in a Jupyter notebook, whose kernel's
  stdin is not a terminal, or with output redirected to a file); its "[A]ll" answer, which used to apply to one loader,
  now switches the process to `"auto"`. `"auto"` used to install the loader's import name with `-U`, even where it is not
  a package name, and did not check pip's exit status; it now installs the declared kornia extra instead of the import
  name (`pip install "kornia[<extra>]"`, for example `kornia[image]`, without `-U`), raises `ImportError` if pip fails
  (without running pip again for that loader), and installs nothing for a loader that declares no extra (`"ask"` does not
  ask for one either). The new `KORNIA_INSTALLATION_MODE` environment variable sets the mode when kornia is imported
  (`raise`, `ask` or `auto`, any case); another value does not stop `import kornia`, but raises a `ValueError` naming the
  variable, the valid modes and the missing package the first time a missing optional package is handled. The undocumented
  `LazyLoader.auto_install` class switch is removed; set the mode to `"auto"` instead. `InstallationMode` members are now
  hashable and compare equal only to their upper-case values (`InstallationMode.ASK == "ASK"`): `==` used to ignore case
  while `!=` did not, so `InstallationMode.ASK == "ask"` and `InstallationMode.ASK != "ask"` were both true. The
  `installation_mode` setter still accepts any case. ([#5240](https://github.com/kornia/kornia/pull/5240))
- The in-place operators of `kornia.core.TensorWrapper`, and so of `Vector2`, `Vector3` and `Scalar` (`+=`, `-=`, `*=`,
  `/=`, `//=`, `%=`, `**=`, `&=`, `|=`, `^=`, `<<=` and `>>=`), now update the wrapped tensor in place and return the same
  wrapper, as they do on a tensor. They used to rebind the name to a new wrapper: `+=`, `-=`, `*=`, `/=` and `//=` with
  any operand, and the others with a tensor operand (with a Python number or another wrapper they raised `TypeError`).
  So an alias kept the old values, and the tensor the wrapper was built from was never modified. A wrapper does not copy
  the tensor it is built from, so `v = Vector3(t); v += 1` now also modifies `t`. An in-place operator now raises
  `RuntimeError` where the tensor's in-place operator raises, where it used to return a new wrapper: on a wrapper of a
  leaf that requires grad, such as `Vector3(nn.Parameter(...))` ("a leaf Variable that requires grad is being used in an
  in-place operation"), for an update that changes the dtype, such as `/=` on an integer wrapper ("result type Float
  can't be cast to the desired output type Long"), and for an update that broadcasts to a larger shape. Write
  `v = v + x` to keep the old behaviour. `@=` still rebinds, as it does on a tensor. ([#5243](https://github.com/kornia/kornia/pull/5243))
- `ONNXExportMixin.to_onnx` (behind `ImageModule.to_onnx`, `kornia.core.ImageSequential.to_onnx` and the model
  exporters) now requests ONNX opset 18, the opset the dynamo exporter builds natively, and documents it. It used to
  request 17: the dynamo exporter, the default of `torch.onnx.export` from torch 2.9, met that only when every operator
  had a down-conversion adapter (a convolution came back at 17, a `GaussianBlur2d` pipeline at 18 after a logged
  conversion failure), and the legacy exporter of older torch returned 17. Pass `opset_version=` to request another one.
  To chain an export with the pre-exported Hub operators (opset 17) in `ONNXSequential`, pass
  `auto_ir_version_conversion=True`; on torch before 2.9, `to_onnx(opset_version=17)` also works. On torch 2.9 and later
  the IR versions already differed, so the conversion was needed before as well.
  The default input shape (`ONNX_DEFAULT_INPUTSHAPE`) is now `[-1, 3, -1, -1]` instead of `[-1, -1, -1, -1]`, so a call
  without `input_shape` exports a fixed 3-channel input with a dynamic batch, height and width: on torch 2.9 and later
  the old default failed for any model with a fixed channel count, and on older torch it declared the channel dimension
  dynamic even where the model fixed it. A channel-agnostic model exported without `input_shape` used to accept any
  channel count and now takes exactly 3. A `pseudo_shape` given without `input_shape` sets the default's fixed entries,
  so `to_onnx(pseudo_shape=[1, 1, 32, 32])` exports a 1-channel input. Pass `input_shape` for a dynamic channel count.
  An explicit `input_shape` whose fixed entries differ from an explicit `pseudo_shape` now raises `ValueError`; the
  `input_shape` entry used to win silently, so the model was traced with a shape other than the one given.
  `ONNXSequential(..., auto_ir_version_conversion=True)` now converts the models to the highest opset among them instead
  of to 17, so it accepts kornia's own opset-18 exports; on some of them, such as a `GaussianBlur2d` pipeline, it used to
  fail with `No Adapter To Version $17 for Pad`. ([#5244](https://github.com/kornia/kornia/pull/5244))
- `motion_blur` and `MotionBlur` now default to `border_type='reflect'`, and `motion_blur3d` and `MotionBlur3D` to
  `'replicate'`; previously all four zero-padded (`'constant'`). The zero padding pulled pixels whose kernel reaches past
  an edge toward `0`: a constant image came back as low as `0.6` at the border for `kernel_size=5` at angle 0 in 2-D and
  `0.667` for `kernel_size=3` at angle `(0, 0, 0)` in 3-D, and lower at other angles and directions (measured down to
  `0.16` in 2-D, for `kernel_size=7` at angle 120, and `0.2` in 3-D). A constant image now stays constant, as it does
  under `filter2d`, `gaussian_blur2d`, `box_blur` and `filter3d`. Pass `border_type='constant'` to keep the old output.
  `'reflect'` needs each spatial axis longer than `kernel_size // 2`, so a 2-D image with an axis of `kernel_size // 2`
  pixels or fewer now raises at the default border instead of being zero-padded; the 3-D default is `'replicate'` so that
  thin volumes (`D = 1`) still work. On torch releases without a float16 CPU `reflection_pad2d` / `replication_pad3d`
  kernel (2.5.1 among them), float16 CPU input to these four filters now raises at the default, where it used to run with
  zero padding; pass `border_type='constant'` (the old behaviour) or `'circular'` (which needs each axis at least
  `kernel_size // 2` long) there. `RandomMotionBlur` and `RandomMotionBlur3D` keep their own `'constant'` default, so
  their output does not change. ([#5246](https://github.com/kornia/kornia/pull/5246))
- `guided_blur` and `GuidedBlur` now raise when `subsample` is not a positive integer, and `GuidedBlur` raises when it is built rather than on its first call: a `BaseError` ("`subsample` must be a positive integer. Got: 0") for `0`, `-1`, `True`, `False` and `numpy.int64(0)`, and a `TypeCheckError` for any other value that is not an integer (`1.0`, `0.5`, `1.5`, `2.0`, a NumPy float or `numpy.bool_`, `"2"`, `None`, a tensor). `TypeCheckError` is a `BaseError`, and since #5233 also a `TypeError`. Before, with 1-channel guidance, `0`, `-1`, `False`, `True`, `1.0`, `0.5`, `numpy.True_`, `numpy.False_` and `numpy.int64(0)` ran as no subsampling, and, on images of 8 x 8 and up, a float above 1 ran as that factor when `floor(floor(H / s) * s)` is `H` and `floor(floor(W / s) * s)` is `W` (`2.0` and `numpy.float64(2.0)` on 8 x 12, `1.5` on 12 x 12, `2.5` on 10 x 10) and failed with a broadcast `RuntimeError` at the final blend otherwise; on a tiny image it failed earlier, in `interpolate` (`1.5` on 1 x 10) or in the reflect `pad` (`2.0` on 2 x 12). With multi-channel guidance, `0`, `-1`, `False`, `numpy.False_` and `numpy.int64(0)` failed in `view` with a `RuntimeError`, every float (`1.0`, `0.5`, `1.5`, `2.0`, `numpy.float64(2.0)`) failed in `view` with a `TypeError` on images of 8 x 8 and up (earlier on a tiny image, as above), and `True` and `numpy.True_` ran as `1`. `"2"` and `None` raised a builtin `TypeError` from `subsample > 1`, and a tensor a `TypeError` from `interpolate`, with either guidance. A call that passed `0`, `-1`, `False`, `True`, `1.0` or `0.5` to mean no subsampling now has to pass `subsample=1`, and one that passed `2.0` has to pass `2`. Positive integers behave as before, NumPy integers included, with one exception: `guided_blur` now continues with `int(subsample)`, so a narrow NumPy integer no longer overflows its own dtype in the arithmetic. On NumPy 2, `numpy.uint8(2)` raised `OverflowError` with `kernel_size=301` (`numpy.int8(2)` from `kernel_size=129`), and with multi-channel guidance failed with an `OverflowError` or a `RuntimeError` from `view` on an image of 256 px or more (128 px for `numpy.int8(2)`); those calls now work. ([#5253](https://github.com/kornia/kornia/pull/5253))
- `Rt_to_matrix4x4`, `camtoworld_graphics_to_vision_Rt`, `camtoworld_vision_to_graphics_Rt`,
  `camtoworld_to_worldtocam_Rt` and `worldtocam_to_camtoworld_Rt`, and `ARKitQTVecs_to_ColmapQTVecs`, which calls two
  of them, now raise `ShapeError` (a `ValueError`) when `R` and `t` have different batch sizes; `disable_checks()` turns
  the check off like the other kornia checks. Previously `Rt_to_matrix4x4`, the two graphics/vision functions and
  `ARKitQTVecs_to_ColmapQTVecs` failed inside `torch.cat` with torch's `RuntimeError`, and the two worldtocam functions
  broadcast a batch-1 argument: `R` of batch 2 with `t` of batch 1 returned a `(2, 3, 3)`, `(2, 3, 1)` pose, and `R` of
  batch 1 with `t` of batch 2 returned `(1, 3, 3)` and `(2, 3, 1)`. Both calls now raise; any other mismatch used to
  raise `RuntimeError` from the matrix product. ([#5265](https://github.com/kornia/kornia/pull/5265))
- `AugmentationSequential` converts each NumPy argument by its data key, and only `input` takes the image conversion.
  A NumPy mask keeps its dtype and values: an `int64` label 3 used to come out as `float32` 3 / 255, and a `uint8`
  0/255 binary mask as `float32` 0/1; they now come out as `int64` 3 and `uint8` 0/255, as the same mask passed as a
  tensor does. A NumPy mask is still read like a NumPy image, `(H, W)`, `(H, W, C)` or `(B, H, W, C)`. NumPy boxes,
  keypoints and class labels, which went through the image conversion and raised, now convert like their tensors, and
  a dictionary input with NumPy values, which raised `AttributeError`, converts too. (#5236) ([#5274](https://github.com/kornia/kornia/pull/5274))
- `rgba_to_rgb` keeps the leading dimensions of its input when `background_color` is a tuple or list: a `(4, H, W)`
  image used to come back as `(1, 3, H, W)` and now returns `(3, H, W)` (#5323). `rgb_to_rgba`, `bgr_to_rgba`,
  `RgbToRgba` and `BgrToRgba` now raise `ValueError` unless a tensor `alpha_val` has the image's shape with one channel,
  `(*, 1, H, W)`. A multi-channel alpha used to be appended as is, so a `(B, 2, H, W)` alpha returned a five-channel
  `(B, 5, H, W)` image; an alpha whose batch or spatial size differed from the image's failed inside `torch.cat` with a
  `RuntimeError`. ([#5296](https://github.com/kornia/kornia/pull/5296))
- `lovasz_softmax_loss`, `LovaszSoftmaxLoss`, `lovasz_hinge_loss` and `LovaszHingeLoss` now compute the Jaccard
  weights and the sum over pixels in the dtype of the prediction, or in float32 for a float16 or bfloat16 prediction,
  and return the loss in the dtype of the prediction, promoted with the dtype of `weight` for the softmax variant
  (#5289). Previously the Jaccard weights took `torch.get_default_dtype()`: a float16 or bfloat16 prediction returned a
  float32 loss, a float32 prediction under a float64 default returned float64, and a float64 prediction had float32
  rounding in its weights and gradient. A `weight` wider than the loss was cast down to it, so a float32 prediction
  with a float64 `weight` returned float32; it now returns float64, as `dice_loss` does. Float32 predictions with no
  `weight` or a float32 `weight` under the default float32 are unchanged. ([#5303](https://github.com/kornia/kornia/pull/5303))
- `ssim`, `ssim3d`, `SSIM` and `SSIM3D` now compute integer images in float32 and return a float32 map, and
  `ssim_loss`, `ssim3d_loss`, `SSIMLoss` and `SSIM3DLoss` inherit the fix (#5297). Previously the fractional Gaussian
  weights truncated to zero, so the result did not depend on the images: the map came back in the integer dtype, all 1
  for `max_val=255.0` and all 0 for the default `max_val=1.0`, and each loss returned the same constant for every pair.
  Pixel values are not rescaled, and floating-point inputs are unchanged. ([#5304](https://github.com/kornia/kornia/pull/5304))
- `image_histogram2d` now raises `ShapeError` (a `ValueError`) for an image that is not `(H, W)`, `(C, H, W)` or `(B, C, H, W)`. A rank-0 or rank-1 image used to return a histogram with spurious batch and channel axes, of shape `(1, 1, n_bins)`, and a rank-5 or higher image raised a `RuntimeError` from inside `permute`. ([#5320](https://github.com/kornia/kornia/pull/5320))
- `zca_mean`, `zca_whiten` and `ZCAWhitening.fit` now raise `ValueError` when the sample dimension has fewer than two entries with `unbiased=True` (the default), or none with `unbiased=False`. Previously the covariance was divided by zero, and the result was NaNs, an empty `zca_whiten` output, or a `torch.linalg.LinAlgError` from the SVD, depending on the input shape. Whitening one sample with `unbiased=False` is still supported and returns zeros. ([#5332](https://github.com/kornia/kornia/pull/5332))
- `solve_quartic` now solves a row as a cubic only when its leading coefficient is below the tolerance and the
  scale-invariant root bound `max(|b/a|, |c/a|^(1/2), |d/a|^(1/3), |e/a|^(1/4))` exceeds `1 / tol`, so a quartic
  whose roots are all below `1 / (4 * tol)` in magnitude is solved as a quartic at every scale. Rows such as
  `[1e-7, 0, 0, 0, -1]` previously returned `[0, 0, 0, 0]` and now return their real roots `±56.23`;
  `(x-50)(x-60)(x-70)(x-80)` scaled by `2^-21` in float32 now returns all four roots instead of a single root
  near 31.5. (#4954) ([#5334](https://github.com/kornia/kornia/pull/5334))
- `ApplyColorMap` registers its palette as the buffer `colors`, so `.to(device, dtype)`, `.half()` and `.double()`
  move and convert it with the module, and `state_dict()` saves it. Previously the palette stayed where its `ColorMap`
  was built: forward on a module moved to MPS or CUDA raised a device-mismatch error, a converted module kept returning
  the palette's dtype (float32 after `.half()` or `.double()`), and `state_dict()` was empty. The output now follows
  the module's dtype, and `state_dict()` gains the `colors` key. A state dict saved by an earlier release still loads
  with `strict=True` and keeps the palette the module was built with; one saved by this release carries `colors` and
  fails `strict=True` on an earlier kornia, which does not know the key. `module.colormap` still holds the `ColorMap`
  as passed and is not moved by `.to()`. (#5317) ([#5335](https://github.com/kornia/kornia/pull/5335))
- `integral_tensor` and `IntegralTensor` now raise a kornia error for a repeated axis, including a negative alias of a positive one such as `(1, -1)` on a rank-2 input, and for an axis outside the input's rank. A repeated axis used to be integrated twice (`integral_tensor(torch.ones(2, 2), (1, 1))` returned `[[1, 3], [1, 3]]`), and an axis outside the rank raised `IndexError` from `cumsum`. Distinct axes are accepted in any order. ([#5336](https://github.com/kornia/kornia/pull/5336))
- `Boxes.from_tensor` with `mode="vertices"` or `mode="vertices_plus"`, and `VideoBoxes.from_tensor`, now reject a
  non-finite vertex coordinate when `validate_boxes=True` (the default). Before, the vertex modes ignored
  `validate_boxes` and stored `NaN` and `inf` vertices; pass `validate_boxes=False` to keep them. The check is
  `torch._assert_async`, so it raises `RuntimeError` (the `'xy*'` modes keep raising `ValueError`), runs on CPU and
  CUDA, is skipped on MPS, and keeps the vertex import, and with it `AugmentationSequential` with `"bbox"` data,
  compilable with `fullgraph=True`. The vertex modes still take no extent or shape check, so the rotated, sheared and
  projective quadrilaterals of `transform_boxes` import unchanged. ([#5337](https://github.com/kornia/kornia/pull/5337))
- `apply_colormap` and `ApplyColorMap` map the top of the input range to the last palette color: `1.0` for unit-range
  input and `255` for integer or byte-range input. Previously, with three or more palette colors, the top of the range
  selected the second-to-last color, so no input in the documented range reached the last one: the palette
  `[0, 0.25, 0.75, 1]` mapped `1.0` to `0.75`, and now maps it to `1`. Every other input keeps its color, so with
  `num_colors=256` a `uint8` value `v` now selects palette color `v` for every `v` (`255` used to select color `254`).
  (#5307) ([#5339](https://github.com/kornia/kornia/pull/5339))
- `rgb_to_raw` and `RgbToRaw` now raise `ValueError` for a `cfa` that is not a `CFA` member, as
  `raw_to_rgb` and `raw_to_rgb_2x2_downscaled` already did. Previously any other value, such as a
  string, an integer or `None`, silently returned the green channel as the mosaic. ([#5340](https://github.com/kornia/kornia/pull/5340))
- `RgbToHsv` now defaults to `eps=1e-8`, the default of `rgb_to_hsv`, so the module and the function
  return the same saturation. The module used to default to `eps=1e-6`, which lowered the saturation of
  dark pixels: a pure red pixel of value `1e-4` had saturation `0.9901` through the module and
  `0.9999` through the function; both now give `0.9999`. Pass `RgbToHsv(eps=1e-6)` to keep the old
  module output. ([#5343](https://github.com/kornia/kornia/pull/5343))
- `adjust_hue_raw` now wraps the shifted hue into [0, 2π), matching `rgb_to_hsv`. A negative sum used to stay negative: hue 0 shifted by -0.5 returned -0.5, and now returns 2π - 0.5. A sum that rounds up to 2π returns 0. `adjust_hue` output changes only by rounding. ([#5345](https://github.com/kornia/kornia/pull/5345))
- `histogram`, `histogram2d` and `image_histogram2d` now reject a bandwidth that is zero, negative or not finite, and `image_histogram2d` also rejects `n_bins < 1` and an empty or non-finite `[min, max]` range when it builds the centers or the bandwidth from them. These inputs used to return NaNs or meaningless values, or a low-level error (`n_bins=0` raised `ZeroDivisionError`); `histogram` and `histogram2d` returned the same result for a negative bandwidth as for its absolute value. `histogram` and `histogram2d` now raise `RuntimeError` from an asynchronous check, like the other tensor-parameter checks in `kornia.enhance` (skipped on MPS); `image_histogram2d` raises `ValueError`. `image_histogram2d` now treats its documented `bandwidth=-1` as the automatic bandwidth, where it used to apply -1 as a literal negative bandwidth, and an empty `centers` tensor as automatic centers, where it used to return zero bins. ([#5347](https://github.com/kornia/kornia/pull/5347))
- `filter2d`, `filter3d`, `filter2d_separable` and `fft_conv` now raise a `BaseError` naming both sizes when the kernel
  batch is neither 1 nor the input batch (#5154). Inputs that used to return a value now raise: `filter2d`,
  `filter2d_separable` and `filter3d` applied a kernel batch that divides the input batch cyclically, sample `i`
  filtered with kernel `i % Bk`, and `fft_conv` broadcast one input sample over a batch of kernels, one output per
  kernel. The same holds for the blurs that build one kernel per sample: `gaussian_blur2d` with a tensor `sigma`, and
  `motion_blur` and `motion_blur3d` with tensor `angle` and `direction`, whose batch divides the input batch. The other
  kernel batches used to raise torch's `RuntimeError` and now raise the `BaseError`, which is not a `RuntimeError`. The
  documented `(1, kH, kW)` and `(B, kH, kW)` kernels are unchanged. ([#5359](https://github.com/kornia/kornia/pull/5359))
- `shift_rgb` now raises `ShapeError` for input that is not `(B, 3, H, W)`. It used to accept any input with three
  channels at axis -3 and broadcast the `(B, 3, 1, 1)` shifts against its leading axes without an error: a
  `(B, T, 3, H, W)` input received the per-batch shifts along `T` instead of `B`, and a `(3, H, W)` input came back
  as `(B, 3, H, W)`, one copy per shift. `RandomRGBShift` always passes `(B, 3, H, W)` and is unaffected. ([#5360](https://github.com/kornia/kornia/pull/5360))
- `image_histogram2d` now keeps the bin axis for rank-2 `(H, W)` input with a single bin (`n_bins=1` or one
  explicit center): the histogram and the PDF have shape `(1,)`, the documented `(bins,)`. They used to be 0-d
  tensors, because the synthetic batch and channel axes were removed with an unrestricted `squeeze()` that also
  dropped the length-one bin axis. Rank-3 and rank-4 input and rank-2 input with more than one bin are unchanged. ([#5366](https://github.com/kornia/kornia/pull/5366))
- `get_gaussian_discrete_kernel1d` now returns `kernel_size` taps and rejects an even `kernel_size` with
  `force_even=True` (#5158). Previously it returned `2 * (kernel_size // 2) + 1` taps and never fewer than 3:
  `kernel_size=1` gave 3 taps and `kernel_size=4` with `force_even=True` gave 5. `kernel_size=1` is now the single tap
  `[1.0]`, and an even size raises a `BaseError`, because the discrete Gaussian is defined at integer offsets from the
  centre tap. `get_gaussian_erf_kernel1d` with `force_even=True` now centres an even kernel on the middle of the window,
  `(kernel_size - 1) / 2`, as `get_gaussian_kernel1d` does; previously it was centred on tap `kernel_size // 2`, so
  `(4, 1.0)` gave `[0.0654, 0.2608, 0.4131, 0.2608]` and now gives `[0.1424, 0.3576, 0.3576, 0.1424]`. Odd sizes above 1
  are unchanged in both builders. ([#5376](https://github.com/kornia/kornia/pull/5376))
- `filter2d` and `fft_conv` now pad for `padding='SAME'`, and for every other spelling of `'same'` (#5156), and so do
  `filter2d_separable`, `correlate2d` and `convolve2d`, which pass `padding` to `filter2d`. Previously the padding check
  accepted any spelling but only `'same'` took the padded branch, so `padding='SAME'` silently returned the
  `'valid'`-size output: a 3×3 kernel on a 5×7 image gave 3×5 and now gives 5×7, the output of `padding='same'`.
  `'VALID'` still crops, and lower-case calls are unchanged. ([#5379](https://github.com/kornia/kornia/pull/5379))
- `denormalize` and `Denormalize` used to accept a `(C,)` mean or std whose length matched dimension -3 of a rank-3 or rank-5 input instead of its channel count, and broadcast a 1-channel input to C channels (a `(2, 1, 4, 2, 2)` input with 4-element statistics returned a `(2, 4, 4, 2, 2)` tensor). They now raise `ValueError` for that input, as `normalize` does. ([#5384](https://github.com/kornia/kornia/pull/5384))
- `bilateral_blur`, `joint_bilateral_blur`, `BilateralBlur` and `JointBilateralBlur` now raise a kornia error for a
  `sigma_color` that is not positive, and `edge_aware_blur_pool2d` and `EdgeAwareBlurPool2D` now require an
  `edge_threshold` greater than 1 (#5169). Previously a tensor `sigma_color` of zeros returned an all-NaN image and a
  negative `sigma_color` blurred exactly like its absolute value. An `edge_threshold` below 1 marked every pixel as an
  edge and returned the input unchanged, and exactly 1 counted any difference between pixels 4 apart as an edge. Under
  `torch.compile` a tensor `sigma_color` is not checked, like the tensor `sigma` of `gaussian_blur2d`. Positive
  `sigma_color` values and thresholds above 1, including the default 1.25, are unchanged. ([#5387](https://github.com/kornia/kornia/pull/5387))
- `Sepia()` and `sepia_from_rgb` with the default `rescale=True` now divide all three output channels by one maximum per
  image, taken over channels and pixels, so the sepia tint survives the rescale. They used to divide each channel by its
  own spatial maximum, which made the output almost grey: a uniform 0.5 grey image used to give `(1.0, 1.0, 1.0)` and now
  gives `(1.0, 0.8905, 0.6936)`. Default outputs change for every image; `rescale=False` is unchanged. ([#5388](https://github.com/kornia/kornia/pull/5388))
- `AugmentationSequential` converts a mask given as a PIL image or an image file path like the same mask passed as a
  NumPy array, keeping its dtype and label values. A mode `L` label 3 used to come out as `float32` 3 / 255 and now
  comes out as `uint8` 3; a palette (mode `P`) mask used to be replaced by its RGB colours scaled to `[0, 1]`,
  `(1, 3, H, W)`, and now keeps its indices, `(1, 1, H, W)`. A file-path mask, and a PIL mask in a dictionary input,
  raised `AttributeError` and now convert. A mask whose height and width differ from the image's, or whose batch size
  is neither 1 nor the image's, now raises `ValueError`; it used to pass unchecked, so a NumPy `(B, H, W)` label batch
  was silently read as one `(H, W, C)` image. (#5291) ([#5390](https://github.com/kornia/kornia/pull/5390))
- A nested `AugmentationSequential` now records its transformation matrix under its own `transformation_matrix_mode`
  when it runs inside another container. A nested container built with `transformation_matrix_mode="rigid"` therefore
  raises `RuntimeError` for a non-rigid child, as it does when called directly; it used to run without the check. (#4476) ([#5391](https://github.com/kornia/kornia/pull/5391))
- The deprecated `gaussian_blur2d_t`, `get_gaussian_kernel1d_t`, `get_gaussian_kernel2d_t`, and `get_gaussian_kernel3d_t` aliases have been removed; use `gaussian_blur2d`, `get_gaussian_kernel1d`, `get_gaussian_kernel2d`, and `get_gaussian_kernel3d` instead. ([#5400](https://github.com/kornia/kornia/pull/5400))
- `otsu_threshold` and `OtsuThreshold` now histogram each image/channel plane over its own range, so a plane's threshold
  no longer depends on the other images and channels in the call (#5172). Previously one range was shared by the whole
  input. The default histogram path now returns the upper edge of the selected bin, `min + (t + 1)·(max − min)/nbins`.
  Previously it read `linspace(min, max, nbins)`, a threshold up to one bin above the split, which at `nbins=2` was
  always the maximum, so nothing was kept. For an integer input the threshold is now the largest integer below that
  edge. Previously it was truncated toward zero, so a negative edge rounded up and an edge on an integer stayed put,
  and the level just above the split was dropped from the foreground. A constant plane now uses its own value as the
  threshold, so none of its pixels is foreground. Previously its threshold was 0. The `slow_and_differentiable=True`
  path also uses each plane's own range and the same integer rounding. ([#5402](https://github.com/kornia/kornia/pull/5402))
- `kornia.enhance.denormalize` and `Denormalize` return torch's default floating dtype (float32 unless changed with `torch.set_default_dtype`) for an integer or bool input, as `normalize` already did. They used to compute in and return the input's integer dtype, which was exact for integral statistics that did not overflow: `denormalize` of a uint8 `[0, 50, 100]` with `mean=10.0, std=2.0` returned uint8 `[10, 110, 210]` and now returns float32 `[10.0, 110.0, 210.0]`. Both functions now convert an integer input before applying `mean` and `std`, so under the default float32 an integer value above 2**24 is rounded first. Cast the result back, for example with `.round().clamp(0, 255).to(torch.uint8)`, where integer output is needed. ([#5407](https://github.com/kornia/kornia/pull/5407))
- `apply_colormap` and `ApplyColorMap` no longer choose the input range from the maximum over the whole tensor. Integer
  inputs always use `[0, 255]`: previously an integer tensor whose maximum was at most 1, such as a `{0, 1}` mask, was
  divided by 1 and mapped to the full palette, and now `1` maps to the palette entry for 1/255. Floating-point inputs
  choose `[0, 1]` or `[0, 255]` separately for each sample and channel: previously one sample or channel above 1 made
  every other sample and channel of the tensor use `[0, 255]`. Refs #5306. ([#5413](https://github.com/kornia/kornia/pull/5413))
- `So2.hat` / `So2.vee` and `Se2.hat` / `Se2.vee` now use the standard Lie algebra generators. `So2.hat(theta)` used to
  return the symmetric `[[0, theta], [theta, 0]]` and `So2.vee` read its `[0, 1]` entry; it now returns
  `[[0, -theta], [theta, 0]]` and `vee` reads `[1, 0]`. `Se2.hat(v)` used to put `(v_x, v_y)` in the bottom row
  below that symmetric block; it now returns `[[0, -theta, v_x], [theta, 0, v_y], [0, 0, 0]]`, and `Se2.vee` reads the
  translation from the last column. `vee(hat(v))` is still the identity, and `matrix_exp(hat(v))` now equals
  `exp(v).matrix()`, as it already did for `So3` / `Se3`. Code that builds or reads these matrices in the old layout
  gets different values. (#5414) ([#5414](https://github.com/kornia/kornia/pull/5414))
- :class:`~kornia.geometry.camera.pinhole.PinholeCamera` ``project`` and ``unproject`` now build their projection
  matrix from the top-left 3x3 block of ``intrinsics`` only (#4771). A 3x3 ``K`` zero-padded to 4x4
  (``intrinsics[3, 3] = 0``) used to project while ``unproject`` raised ``LinAlgError`` on the singular 4x4 product;
  it now round-trips exactly like its homogeneous embedding. The other entries of the 4x4 no longer enter either
  method: before, a non-zero ``intrinsics[0, 3]`` shifted every projected ``u`` by ``intrinsics[0, 3] / z`` (with
  ``K = [[100, 0, 4], [0, 100, 3], [0, 0, 1]]`` and ``intrinsics[0, 3] = 10``, the point ``(1, 2, 4)`` projected to
  ``u = 31.5``, now ``29``), and ``intrinsics[3, 3] = 2`` doubled every point ``unproject`` returned. Intrinsics in
  the documented homogeneous form give the same results as before, bit for bit on CPU. ([#5415](https://github.com/kornia/kornia/pull/5415))
- `otsu_threshold` and `OtsuThreshold` with `slow_and_differentiable=True` now return a threshold with a gradient with
  respect to the input, and return different threshold values:

  - The threshold now has a gradient. Its value is the hard Otsu split of the path's histogram, and its gradient is a
    straight-through surrogate: the gradient of a soft-argmax over the between-class variance curve of a wider kernel
    density estimate, with a standard deviation of 0.5 bin, at a temperature relative to the curve, so it does not depend
    on the intensity scale of the input. The threshold used to have no gradient. The thresholded image is still
    `x * (x > threshold)`, whose gradient is that mask.
  - The histogram is now the mass that a Gaussian kernel density estimate with a standard deviation of 0.1 bin puts in
    each of the `nbins` bins, so every pixel contributes its full mass. The estimate used to be sampled at `nbins` points
    with a fixed bandwidth of `1e-3` in input units, which missed every pixel more than a few `1e-3` away from a sample
    point, most pixels at small `nbins`. The thresholds change accordingly, and are usually within one bin of the
    default path's.
  - The threshold is now read with the default path's bin-edge convention: the upper edge of the selected bin among
    `nbins` equal bins over the plane's range, or for an integer input the largest integer below that edge. It used to
    be read from the `nbins` sample points `linspace(min, max, nbins)`, up to one bin above the split. As on the default path,
    a floating-point threshold is kept at or above the largest background pixel and below the smallest foreground
    pixel, so `x > threshold` reproduces the split.
  - Of splits that give the same partition, the lowest is now taken on every device: a bin counts as non-empty only
    above 1 % of one pixel's mass. The kernel tails used to give empty bins a tiny positive mass, the rounding of the
    cumulative sums chose between equivalent splits, and CPU and MPS could return different thresholds for the same
    input.
  - The path no longer calls `.item()`, so it supports `torch.export` and compiles with `fullgraph=True`.
  - The slow path now computes its surrogate on the plane shifted by its minimum and divided by its range, or by its
    largest magnitude when the range overflows, and cancels this normalization from its first-order gradient. Higher-order derivatives through this normalization do not represent the original-unit soft threshold.
    Finite inputs near the dtype limits or with tiny ranges keep finite thresholds and gradients; hard edges whose range
    would overflow or whose bin width would underflow are reconstructed in bounded units as well.

  ([#5426](https://github.com/kornia/kornia/pull/5426))
- `bilateral_blur`, `joint_bilateral_blur`, `BilateralBlur` and `JointBilateralBlur` now raise a kornia error when an
  input of batch 1 is given a tensor `sigma_color` or `sigma_space` with a larger batch (#5430). Previously the
  broadcast succeeded silently and returned a batch as large as the sigma batch, for example 3 images from 1 input and 3
  `sigma_color` values. A sigma batch of 1, or equal to the input batch, is unchanged. ([#5432](https://github.com/kornia/kornia/pull/5432))
- The default (`slow_and_differentiable=False`) path of `otsu_threshold` and `OtsuThreshold` returns different
  thresholds, and for floating-point input sometimes different masks:

  - Of the splits across a run of empty histogram bins, which all give the same classes, the lowest is now taken on
    every device. On MPS with torch 2.14, cumulative-sum rounding used to pick a later split, so the threshold differed
    from the CPU's (for example 0.4766 instead of 0.4000) while the mask was the same.
  - A floating-point threshold is now kept at or above the largest pixel of the background bins and below the smallest
    pixel of the foreground bins, so `x > threshold` reproduces the histogram split. The threshold used to be the bin
    edge itself, in the input dtype: a pixel exactly on the edge, or on a half-precision edge that rounded up onto it,
    was dropped from the foreground although the histogram counted it there, and a pixel the histogram counted in the
    background could lie above the interpolated edge and be kept. Thresholds that already separated the classes stay at
    the bin edge, although the batched edge computation can differ from the previous per-plane `torch.linspace` in the
    last bit (seen on MPS in float32). On MPS, a zero foreground boundary in float32 or bfloat16 gives the threshold
    `-torch.finfo(dtype).tiny`, because MPS comparisons flush subnormal values to zero.
  - A plane containing NaN or an infinite value used to raise `RuntimeError` from `torch.histc`. It now returns a NaN
    threshold and an all-background mask for that plane; the other planes are unaffected.

  ([#5441](https://github.com/kornia/kornia/pull/5441))
- `Quaternion` now registers plain tensor data as a persistent buffer, preserving autograd history
  and explicitly supplied `nn.Parameter` inputs at construction. Previously, plain tensor rotations
  were omitted from module state; `Quaternion`, `So3` and `Se3` now save and restore them
  and follow enclosing module device/dtype conversions. This adds `_data`, `_q._data`
  and `_rotation._q._data` checkpoint keys, respectively. Older checkpoints missing these
  keys now fail with `strict=True`; `strict=False` reports the missing keys and retains
  the target's initialized rotation, which must be supplied separately if needed. ([#5446](https://github.com/kornia/kornia/pull/5446))
- `ImageModule`, `ImageSequential` and `AugmentationSequential` with parameters or buffers now convert NumPy, PIL and
  image-path inputs onto the device of their first parameter (or first buffer) and the dtype of their first floating
  parameter or buffer. Previously a converted image was always created on the CPU, in the default dtype for an integer
  image and in its own dtype for a floating one: a module moved to CUDA or MPS raised a device-mismatch error, or
  returned a CPU output when its forward did not touch its state, and a `float64` NumPy image fed to a `float32` module
  gave a `float64` output where it now gives `float32`. NumPy masks, keypoints and boxes given to `AugmentationSequential`
  move to the same device and keep their dtype. Stateless modules, tensor inputs and the standalone mixin are unchanged. ([#5451](https://github.com/kornia/kornia/pull/5451))
- `Resize`, `LongestMaxSize`, `SmallestMaxSize` and slice-mode `RandomResizedCrop` with `resample="nearest"` now resize with PyTorch's `nearest-exact` grid, which copies source column `floor((u + 0.5) * W_in / W_out)` into output column `u`, and record the matching half-pixel `transform_matrix`, `x' = (x + 0.5) * W_out / W_in - 0.5` (likewise for `y`). Previously they copied the legacy `floor(u * W_in / W_out)` column and recorded a corner-to-corner matrix that the image did not follow, so keypoints and boxes landed up to one output pixel away from the pixels they belong to (#5293). Nearest-resized images and the recorded matrices change. So do masks resized by these augmentations in `AugmentationSequential`, which resamples masks with nearest even when images use bilinear. ([#5458](https://github.com/kornia/kornia/pull/5458))
- `safe_inverse_with_mask` and `safe_solve_with_mask` no longer turn the gradient NaN when one matrix in the batch is singular. Validity is decided on a detached pass and the invalid matrices are replaced by the identity before inverting or solving. An invalid row used to hold NaN in eager mode (a finite adjugate under tracing or export); it now holds the identity inverse (or `B` for the solve) in every mode. A float16 inverse that overflows is now reported invalid instead of `inf` with `mask=True`. `find_homography_dlt`, `find_homography_dlt_iterated`, `find_homography_lines_dlt` and `fundamental_from_essential` still return NaN for degenerate input. ([#5459](https://github.com/kornia/kornia/pull/5459))
- `quaternion_exp_to_log` and `euler_from_quaternion` now normalize their input (#3953), matching `quaternion_to_rotation_matrix` and `quaternion_to_axis_angle`, so a rescaled quaternion gives the same result as the unit one. Unit inputs are unchanged to rounding; non-unit inputs that used to return a wrong value without error now return the value of the rotation they stand for. Before, `quaternion_exp_to_log([0.5, 0.5, 0, 0])` returned `[1.0471975511965976, 0, 0]`; it now returns `[0.7853981633974483, 0, 0]`. Before, a scalar part `w > 1` was clamped to `1`, so `quaternion_exp_to_log([2, 1, 0, 0])` returned the origin; it now returns `[0.4636476090008061, 0, 0]` (`[2, 0, 0, 0]` still returns the origin, the log of the identity). Before, `euler_from_quaternion` of `2 * q`, for the `q` of `(0.3, 0.7, 1.1)`, returned `[1.6560585860248003, 1.5707963267948966, 2.1048169977173687]`, a pitch of exactly `pi/2`; it now returns `(0.3, 0.7, 1.1)`. `quaternion_exp_to_log` computes the angle as `atan2(||v||, w)`, which needs no clamp and has a finite derivative at `w = +-1`. ([#5461](https://github.com/kornia/kornia/pull/5461))
- A shape-changing augmentation such as `RandomCrop` or `Resize` given a hand-made `batch_prob` that mixes applied and skipped rows now raises `ValueError`, in `forward` and in `inverse`. Before, `forward` transformed the skipped rows too, and `inverse` failed with a shape mismatch. `kornia.contrib.augmentation_audit.audit` raises the same `ValueError` for such a step; before, it returned a report for it. ([#5473](https://github.com/kornia/kornia/pull/5473))
- `motion_from_essential_choose_solution` now returns a fourth value, the number of correspondences at positive depth in both cameras for the selected pose candidate, with shape `(B,)` (a 0-d tensor for unbatched inputs), as OpenCV's `recoverPose` does (#4879). Previously it returned `(R, t, points3d)`, and when no correspondence passed the depth test (for example with `mask` all `False`) it returned candidate 0 with nothing to tell the caller. It now returns `(R, t, points3d, num_valid)`, and `num_valid == 0` flags that case. Code that unpacks three values, `R, t, X = motion_from_essential_choose_solution(...)`, must unpack four. ([#5475](https://github.com/kornia/kornia/pull/5475))
- :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.intrinsics_inverse`, :class:`~kornia.geometry.depth.DepthWarper`
  and :func:`~kornia.geometry.depth.depth_warp` now read only the top-left 3x3 block of ``intrinsics``, like
  ``project`` and ``unproject`` since #5415 (#5480). A 3x3 ``K`` zero-padded to 4x4 (``intrinsics[3, 3] = 0``) used
  to raise ``LinAlgError`` in all three; ``intrinsics_inverse`` now returns the homogeneous 4x4 embedding of
  ``inv(K)``, and the warp equals the one for the homogeneous embedding. The other entries of the 4x4 no longer
  enter: with ``K = [[100, 0, 4], [0, 100, 3], [0, 0, 1]]``, ``intrinsics[0, 3] = 10`` used to give
  ``intrinsics_inverse()[0, 3] = -0.1`` (now ``0``) and, on the destination camera, shifted every ``DepthWarper``
  sampling position by ``10 / z`` pixels; ``intrinsics[3, 3] = 2`` used to give ``intrinsics_inverse()[3, 3] = 0.5``
  (now ``1``) and, on the source camera, doubled the depth the warp unprojects. Intrinsics in the documented
  homogeneous form give the same results as before, bit for bit on CPU. ([#5485](https://github.com/kornia/kornia/pull/5485))
- `RandomCrop(cropping_mode="slice")` now places bounding boxes and keypoints on the half-pixel grid of the slice resize
  when the requested crop exceeds the padded input along an axis, `x' = (x + 0.5) * s - 0.5` for the stretch `s` (#5481).
  Previously the transform matrix scaled about the origin, `x' = s * x`, so annotations sat `0.5 * (s - 1)` pixels before
  the image content along that axis: 0.132 pixels at a 416 / 329 stretch and 0.5 pixels at a 2x stretch. Axes that fit,
  resample mode and the image itself are unchanged. ([#5487](https://github.com/kornia/kornia/pull/5487))
- `safe_inverse_with_mask` and `safe_solve_with_mask` now decide the singularity of 2x2, 3x3 and 4x4 matrices by one
  rule in eager mode and under capture alike, the closed-form determinant against the permanent of the absolute values,
  `|det A| <= 8 n eps perm |A|` (#5476). Previously the mask came from LAPACK's `info` in eager mode and from `det != 0`
  under capture, so a 3x3 with two equal rows was valid on torch 2.5.1 eager and under `torch.jit.trace` and invalid on
  torch 2.14 eager. A matrix whose determinant is rounding noise is now invalid everywhere, whatever its scale, and
  `_torch_inverse_cast` documents that it does not check singular input. ([#5488](https://github.com/kornia/kornia/pull/5488))
- `get_motion_kernel2d` with `mode="nearest"`, and with it `motion_blur`, `MotionBlur` and `RandomMotionBlur`, used to let roundoff pick the tap where the rotated line falls exactly between two taps (some multiples of 30 degrees), so the kernel changed with the dtype, the device, the torch version, `torch.compile` and a full turn added to the angle. The tie now goes to the even coordinate, as `grid_sample` breaks it with exact arithmetic, so at those angles the CPU float32 kernel can differ from the one it used to be: `get_motion_kernel2d(5, 330.0, 0.3)` gives other taps than before. A nearest kernel still has a zero gradient in the angle. ([#5497](https://github.com/kornia/kornia/pull/5497))
- `get_motion_kernel3d` with `mode="nearest"`, and with it `motion_blur3d`, `MotionBlur3D` and `RandomMotionBlur3D`, used
  to let roundoff pick the tap where the rotated line falls exactly between two taps (a pitch or roll that is a multiple
  of 30 degrees), so the kernel changed with the dtype: float32 and float64 built different kernels at 6 of the 1000
  angles of a 10-value grid, by up to 0.16 (#5510). The tie now goes to the even coordinate, as `grid_sample` breaks it
  with exact arithmetic, so at those angles the float32 kernel can differ from the one it used to be:
  `get_motion_kernel3d(5, (0.0, 330.0, 0.0), 0.3)` keeps 7 taps where it used to keep 5. A nearest kernel still has a
  zero gradient in the angle. ([#5516](https://github.com/kornia/kornia/pull/5516))
- `VideoSequential(same_on_frame=False)` now draws separate parameters for every frame in a nested container, such
  as an `ImageSequential`, as it already did for augmentations placed in it directly. It used to draw one set per
  video. With a single video, every frame got that same set. With several videos, the call raised, for example
  `grid_sampler(): expected grid and input to have same batch size` for `RandomAffine`, or a flip gave every frame of
  every video the same result. A nested `RandomCrop` raised `IndexError` in both cases. ([#5518](https://github.com/kornia/kornia/pull/5518))
- `bilateral_blur`, `joint_bilateral_blur`, `BilateralBlur` and `JointBilateralBlur` used to return a wider dtype than a floating input when a tensor `sigma_space` or the `guidance` was wider, for example float64 for a float32 image with a float64 `sigma_space`, or float32 for a float16 image with a default float32 `sigma_space` tensor. Both are now cast to the input's dtype, as a tensor `sigma_color` already was, so the output keeps the input's dtype. ([#5524](https://github.com/kornia/kornia/pull/5524))
- `psnr`, `psnr_loss` and `PSNRLoss` now compute integer and bool images in float32, as `ssim` does, and compare the
  two images in their promoted dtype (#5536). Two integer or bool images give the value of their float32 copies as a
  float32 tensor. Previously they raised torch's dtype error, except on MPS with torch 2.5.1, where `uint8`, `int16`,
  `int32` and `bool` pairs returned wrong values and `int64` aborted the process. An integer image paired with a
  float16 or bfloat16 image now gives a float32 result; previously the result was in the half dtype. Two images of
  different dtypes no longer abort the process on MPS. Pixel values are not rescaled, and two floating-point images of
  one dtype are unchanged. ([#5552](https://github.com/kornia/kornia/pull/5552))
- `ssim`, `ssim3d`, `SSIM`, `SSIM3D`, `ssim_loss`, `SSIMLoss`, `ssim3d_loss` and `SSIM3DLoss` now read `padding`
  case-insensitively and reject any other value, as the filters do (#5537). Previously every value other than `'valid'`
  silently took the `'same'` branch. `padding='VALID'` or `'Valid'` returned the uncropped map, so a 5-pixel window on
  a 13×17 image gave 13×17 and now gives 9×13, the output of `padding='valid'`, and the losses now use that cropped
  map. A misspelling such as `'full'` or `'bogus'` returned the `'same'` map and now raises kornia's
  `Invalid padding mode` error, which names the accepted options. `'same'`, `'valid'` and the case variants of `'same'`
  are unchanged. ([#5553](https://github.com/kornia/kornia/pull/5553))
- `kl_div_loss_2d` and `js_div_loss_2d` now raise `NotImplementedError` for a `reduction` outside `'none'`, `'mean'`
  and `'sum'`, and check that `pred` and `target` are 4-D `(B, N, H, W)` tensors of the same shape (#5535). Before, any
  other reduction, torch's `'batchmean'` and `None` included, silently returned the `'sum'`, B · N times the `'mean'`
  value; kornia's `'mean'` is torch's `'batchmean'` on the `(B · N, H · W)` reshape. Before, `kl_div_loss_2d` also read
  a `pred` of another shape with the same number of elements, such as one with H and W swapped, in the layout of
  `target` and returned a value; it now raises `BaseError`. Shape mismatches that already failed now raise kornia's
  `BaseError`, or `ShapeError` for an input that is not 4-D, instead of torch's `RuntimeError` or a bare `ValueError`. ([#5555](https://github.com/kornia/kornia/pull/5555))
- `mean_average_precision` now computes the average precision of integer boxes in float32, and that of two box sets of
  different dtypes in their promoted floating dtype, with integer boxes counting as float32 as in `mean_iou_bbox`
  (#5551). Previously integer boxes raised torch's `mean(): could not infer output dtype`, and box sets of two floating
  dtypes kept the TP/FP counts and the result in the prediction dtype and the precisions in the ground-truth dtype:
  float16 predictions with float32 ground truth returned 0.8486 in float16 where float32 boxes give 0.8485, and they
  now return that float32 value. Boxes of one floating dtype give the same result as before. A mismatch of the five list
  lengths, or of the box, label and score counts, now raises kornia's `BaseError` naming the lengths or counts instead
  of a message-less `AssertionError`, so code that catches `AssertionError` there must catch `BaseError` instead. ([#5556](https://github.com/kornia/kornia/pull/5556))
- `ssim3d`, `SSIM3D`, `ssim3d_loss` and `SSIM3DLoss` now reflect the volume at its faces for `padding='same'`, as
  `ssim` does for images (#5534). Previously they replicated the border, so a stack of identical slices did not
  reproduce the 2-D map near the faces, by up to 0.61 on random data. `'same'` results change within `window_size // 2`
  voxels of every face; the border does not reach `'valid'` results. A volume whose depth, height or width is
  `window_size // 2` or less used to return a `'same'` map and now raises torch's padding error, as `ssim` does for such
  images. Complex volumes, for which `ssim3d` returned a complex map, now raise the error that `ssim` raises for complex
  images, because the window is built in the input dtype: a `NotImplementedError`, or a plain `RuntimeError` on torch
  2.5.1. ([#5560](https://github.com/kornia/kornia/pull/5560))
- `dice_loss`, `tversky_loss`, `DiceLoss` and `TverskyLoss` now require a `target` of shape `(B, H, W)` for a `pred` of
  shape `(B, C, H, W)` and raise `ValueError` otherwise (#5544). They used to compare only `H` and `W`, and some
  mismatched targets returned a value: with a `(1, H, W)` target for `B > 1`, `tversky_loss` returned the loss of sample
  0 alone and `dice_loss` broadcast the target over the batch; with `B = 1`, `dice_loss` averaged the prediction's loss
  over a larger target batch and returned a wrong loss for a `(1, 1, H, W)` target. Callers that relied on the broadcast
  must expand the target themselves. Most other mismatches ended in torch's `RuntimeError`; they now raise `ValueError`
  too, which `except RuntimeError` does not catch. ([#5561](https://github.com/kornia/kornia/pull/5561))
- `HausdorffERLoss3D` now validates its target as `HausdorffERLoss` does: `target` must be `torch.long` with every label
  in `[0, C)`, otherwise it raises `ValueError` naming the accepted range (#5545). The 3-D loss used to check only that
  no label exceeded `C`. It returned a loss for float targets and for integer targets other than `torch.long`, such as
  `int32` or `uint8` (the right loss when every label was a whole number in `[0, C)`), and it silently dropped voxels
  labelled `-1`, `C` or a fractional value from every class map. These targets now raise. A label above `C` used to
  raise `ValueError("Invalid target value.")` and now raises the range message. ([#5562](https://github.com/kornia/kornia/pull/5562))
- `dice_loss(average="micro", weight=w)` and `DiceLoss(weight=w)` now weight the intersection once, computing the weighted Dice `2 Σ w p t / Σ w (p + t)`. They used to square the weight in the intersection, so a uniform weight scaled the Dice coefficient (a perfect prediction scored -1 with `weight=2`) and non-uniform weights could push the loss below -1. Weighted `"micro"` losses with weights other than 0 and 1 change value. Unweighted, 0/1-weighted and `"macro"` losses are unchanged. ([#5564](https://github.com/kornia/kornia/pull/5564))
- `confusion_matrix` and `mean_iou` now require `pred` and `target` to be integer or `bool` tensors with values in
  `[0, num_classes)`, and form the cell index in `int64` (#5549). Previously the labels were not checked: an
  out-of-range prediction was counted silently in another cell (with 3 classes, a prediction of 3 for target 0 counted
  as target 1 / prediction 0), and it now raises `BaseError` naming the offending range, as an out-of-range target
  does instead of a raw torch error. A list now raises `TypeError` instead of `AttributeError`, and a float tensor
  `BaseError` instead of a raw torch error. Small integer label maps now give the matrix of their `int64` copy. The
  index used to wrap in the label dtype: `uint8` maps from 17 classes, so a perfect `uint8` prediction with 21 classes
  reported an IoU of about 5e-7 for two classes that do not occur, and `int8` maps from 12 classes and `int16` maps
  from 182, into another cell or a raw `bincount` error. The `num_classes` message now says "at least two". ([#5565](https://github.com/kornia/kornia/pull/5565))
- `ssim3d_loss` and `SSIM3DLoss` now compute the documented structural dissimilarity `clamp((1 - SSIM) / 2, 0, 1)`,
  the form `ssim_loss` uses (#5533). Previously they returned the unclamped `1 - SSIM`, twice the 2-D loss on matched
  content and above 1 on anti-correlated content. A caller who wants the old value can reduce
  `1 - kornia.metrics.ssim3d(...)`. ([#5567](https://github.com/kornia/kornia/pull/5567))
- `kornia.metrics.confusion_matrix(normalized=True)` now divides each row (target class) by its sum, so every non-empty
  row sums to 1: the per-class recall, scikit-learn's `normalize='true'`, as in torchnet's `ConfusionMeter`, which the
  code cites (#5550). Previous releases divided each column (predicted class) by its sum, scikit-learn's
  `normalize='pred'`: for targets `[0, 0, 0, 1, 1, 2, 2, 2]` and predictions `[0, 0, 1, 1, 1, 2, 2, 0]` the rows summed
  to 1, 0.6667 and 1.3333, and now each sums to 1. Empty rows stay zero, and the `1e-6` added to each sum is unchanged.
  For the previous column normalization, compute `cm / (cm.sum(dim=1, keepdim=True) + 1e-6)` from the unnormalized
  `(B, K, K)` matrix `cm`. ([#5568](https://github.com/kornia/kornia/pull/5568))
- The mutual information losses (`mutual_information_loss`, `normalized_mutual_information_loss`, their `_2d` and `_3d`
  forms and the `MILossFromRef*` and `NMILossFromRef*` modules) now normalise each signal onto the bin centres
  `0 … num_bins - 1`, so every sample contributes to the histogram and the result no longer depends on the intensity
  polarity (#5546). Previously the signals were normalised onto `[0, num_bins]`, one bin width past the last centre, so
  with the default `MIKernel.xu` the pixels at an image's maximum got no histogram weight, and the other two kernels gave
  them only part of it: in float64 a binary mask had mutual information 2.4e-13 with itself instead of its entropy, and
  normalised mutual information 1.77 instead of 2. Integer images with levels `0 … K` now sit on the bin centres with
  `num_bins=K + 1`, not `num_bins=K`. Every value changes, a little on continuous images and a lot on binary, label or
  saturated images. ([#5569](https://github.com/kornia/kornia/pull/5569))
- `kornia.losses.one_hot` now returns the exact one-hot tensor in the requested `dtype`: `eps` defaults to 0 and is
  applied only when it is non-zero, and a non-zero `eps` with an integer or bool `dtype` raises (#5538). Previously
  every zero was `eps = 1e-6`, each pixel summed to `1 + (C - 1) * eps`, and an integer or bool `dtype` came back as
  float32. `dice_loss` and `focal_loss` build their targets with it, so their values lose the eps terms: the macro
  Dice loss of a perfect prediction of a 256 × 384 image with a one-pixel class is now 0 instead of 0.0234, and the
  non-target entries of `focal_loss(reduction='none')` are 0 instead of `1e-6` times the class term. Code that takes
  a log of the output of `one_hot` now needs its own floor, or can pass `eps` explicitly. `dice_loss` builds its
  target in float32 for float16 and bfloat16 input, so the float16 gradient of a class absent from a sample stays
  finite without the floor, including on the few-pixel images where it was NaN before. ([#5571](https://github.com/kornia/kornia/pull/5571))
- `kornia.image.tensor_to_image` now maps a one-channel `(1, H, W)` tensor to `(H, W)` for every H and W:
  `(1, 1, W)` gives `(1, W)`, `(1, H, 1)` gives `(H, 1)` and `(1, 1, 1)` gives `(1, 1)`, with either `keepdim`. They
  used to come out as the 1-D arrays `(W,)` and `(H,)` and a 0-d array, because the 3-D branch squeezed every unit
  axis, so a 1-row and a 1-column image could not be told apart; code that indexed the 1-D result now gets the 2-D
  array. `kornia.io.write_image`, and `ImageModuleMixIn.save()` through it, now saves a 1-row or 1-column grayscale
  image like any other, where it used to raise a `TypeError` from the encoder (an `IndexError` for `(1, 1, 1)`)
  (#5570). ([#5572](https://github.com/kornia/kornia/pull/5572))
- `ssim` and `SSIM` now raise torch's error for building the Gaussian window in a complex dtype when one image is real
  and the other complex, in either order (#5574): a `NotImplementedError`, or a plain `RuntimeError` on torch 2.5.1.
  Previously a real `img1` with a complex `img2` returned a complex map, while a complex `img1` or two complex images
  raised that error. `ssim_loss` and `SSIMLoss` already raised for such pairs. ([#5578](https://github.com/kornia/kornia/pull/5578))
- `mean_average_precision` now checks in every image that the boxes, labels and, for detections, the scores have one row
  each, and raises `BaseError` naming the image and its counts otherwise (#5580). Previously only the totals over all
  images were compared, so mismatches that cancelled over the images passed the check and the boxes were paired with the
  wrong image, which returned a wrong mAP with no error: in the issue's two-image example 0.5455 instead of an error.
  Input whose counts agree in every image gives the same result as before. ([#5582](https://github.com/kornia/kornia/pull/5582))
- `dice_loss` and `DiceLoss` with `average="macro"` now exclude classes with no
  non-ignored target pixel in each sample, and drop their weights from both the
  numerator and the denominator of the weighted mean. Previously, each absent
  class entered the mean as loss 1, so the loss grew when extra class channels
  were added. Micro averaging and samples containing every class are unchanged.
  Samples with all pixels ignored, or whose present classes all have weight 0,
  keep a loss of 1. ([#5590](https://github.com/kornia/kornia/pull/5590))
- The mutual information losses (`mutual_information_loss`, `normalized_mutual_information_loss`, their `_2d` and `_3d`
  variants and the `MILossFromRef` / `NMILossFromRef` modules) now validate their arguments (#5548). Inputs that used to
  return a value now raise kornia's `BaseError`: an integer mask (an `int64` 0/1 mask indexed positions 0 and 1 and gave
  MI 0); a mask that is not shaped like one sample, `(N,)`, `(H, W)` or `(D, H, W)` (a `(W, H)` mask on `(H, W)` images
  was read in row-major order, a flat `(H·W,)` mask on the `_2d` form was accepted, and a one-element or 0-d `True` mask
  on the flat form meant no mask); masks whose intersection is empty (disjoint masks gave MI 0 and an NMI loss of -1); a
  `num_bins` that is not an `int` of at least 2 (`1` gave 0, and floats such as `2.5` or `16.0` and NumPy integers ran);
  and a negative `window_radius`. The functions raise `ShapeError` when `input` and `target` differ in shape, and the
  `_2d` / `_3d` modules raise `BaseError` when the compared image differs from the reference: a `(W, H)` target against
  an `(H, W)` input used to be flattened first and return a value. Errors that used to come from torch or Python now
  come from kornia too: float masks, empty masks, batched masks, `window_radius=0` (`ZeroDivisionError`) and a size
  mismatch in the `_2d` / `_3d` modules raise `BaseError`, so code that caught the `ValueError` of a batched flat mask
  or of that size mismatch must catch `BaseError`. A `kernel_function` that is not an `MIKernel` member raises the
  documented `ValueError` on every Python version instead of `TypeError` or `AttributeError`. Non-contiguous masks are
  accepted, `uint8` masks keep working without torch's deprecation warning, and `MIKernel.rectangular` keeps the input
  dtype: float64, float16 and bfloat16 inputs returned a float32 loss computed in float32. Calls with boolean masks and
  the `xu` or `truncated_gaussian` kernels return the same values as before, bit for bit. ([#5591](https://github.com/kornia/kornia/pull/5591))
- `RandomTransplantation` inside `AugmentationSequential` now returns a batched `(B, H, W)` mask as `(B, 1, H, W)`, like every other child without `keepdim=True`. Previously it returned the mask as given, so a pipeline whose only child is the transplant, or whose transplant runs after a child with `keepdim=True`, returned `(B, H, W)`. With the container's `keepdim=True` the mask stays `(B, H, W)`. A mask of the wrong rank is now refused with "Every segmentation mask must have one fewer dimension ... than the image" instead of "Every image input must have one additional dimension ...". ([#5592](https://github.com/kornia/kornia/pull/5592))
- `mean_average_precision` now averages only the foreground classes that have ground-truth objects in the evaluated
  images, as COCO does (#5540). Previously a class without ground truth entered the mean as AP 0, with or without
  detections, so mAP depended on `n_classes`: two perfect detections gave 1.0 with `n_classes=3`, 0.667 with 4 and 0.5
  with 5, and now give 1.0 for each. The dictionary entry of such a class is now `-1.0` (undefined, as in COCO) instead
  of `0.0`, and when no foreground class has ground truth the mAP is `-1.0` instead of `0.0` (`nan` with
  `n_classes=1`). A class with ground truth but no detection still contributes AP 0, and class 0 is still excluded as
  background. A predicted or ground-truth label outside `0 <= label < n_classes` used to be skipped silently and now
  raises kornia's `BaseError`. ([#5593](https://github.com/kornia/kornia/pull/5593))
- `tversky_loss` and `TverskyLoss` now compute true positives, false positives and
  false negatives per class and average over classes present in each sample's
  non-ignored target. Previously, pooling across classes made false-positive and
  false-negative mass equal, so `alpha` and `beta` acted only through their sum.
  Coefficient splits with the same sum now give different losses; `alpha=beta=0.5`
  corresponds to macro Dice with exact targets and twice the denominator epsilon,
  rather than micro Dice. Fully ignored samples retain loss 1 and zero gradients.
  Half-precision inputs use float32 through softmax and the ratio to keep gradients
  finite for saturated predictions, while the returned loss keeps the input dtype. ([#5594](https://github.com/kornia/kornia/pull/5594))
- `torch.jit.script(normalize_min_max)` now normalises every `(H, W)` plane of a `(*, C, H, W)` input, as the eager
  function does. Previously scripting compiled only the body written for `(B, C, H, W)`, so a scripted call normalised
  each row of each channel of a `(C, H, W)` input separately, returned an `(H, W)` input filled with `min_val`, and
  normalised an input with more than four dimensions jointly over all but its first two dimensions. Scripted
  `(B, C, H, W)` calls and eager results for non-empty inputs are unchanged. ([#5605](https://github.com/kornia/kornia/pull/5605))
- Indexing a `VideoBoxes` now selects videos instead of rows of its flattened `(B * T, N, 4, 2)` storage.
  `video_boxes[i]` used to hold row `i` of that storage, `(N, 4, 2)` (frame `i % T` of video `i // T`), without
  `temporal_channel_size`; it is now video `i` with all `T` frames, `(T, N, 4, 2)`, and its `to_tensor()` is
  `(1, T, N, 4, 2)`. Slices, integer tensors and boolean masks index videos too, so a boolean mask has length `B`:
  a length-`B * T` mask, which used to select frames, now raises `IndexError`. Index assignment follows the same
  rule: `video_boxes[i] = other` used to write row `i` of the flattened storage and now writes all `T` frames of
  video `i`. ([#5613](https://github.com/kornia/kornia/pull/5613))
- With static p=1.0, p_batch=1.0, a batch_prob put into params is now ignored by every handler, in eager and compiled runs (#5585). The image and transform_matrix already ignored it, but inverse and, for augmentations such as the flips and RandomCrop, the mask, box and keypoint handlers honored it, so a sample it deselected was transformed in forward and left alone by inverse. Pass p < 1 to skip samples by hand. ([#5616](https://github.com/kornia/kornia/pull/5616))
- `AugmentationSequential` now repeats a single tensor mask, `(H, W)`, `(1, H, W)`, `(1, C, H, W)` or one `(C, H, W)` mask, to the batch of a `(B, C, H, W)` image before the first child, so each sample's own parameters apply to its copy and the mask comes back with batch `B`. Previously the result depended on the child: per-sample warps such as `RandomAffine`, `RandomPerspective` and `RandomElasticTransform` raised `grid_sampler(): expected grid and input to have same batch size`, `RandomTransplantation` refused the mask, and flips, crops, resizes and color children such as `ColorJiggle` returned it at batch 1, with flips and crops applying sample 0's parameters only, so it did not match images 1 to `B - 1`. For example, with a batch-2 image a `(1, 1, H, W)` mask through `RandomHorizontalFlip` used to come back as `(1, 1, H, W)` and now comes back as `(2, 1, H, W)`; under `keepdim=True` a `(1, H, W)` mask used to come back as `(1, H, W)` and an `(H, W)` mask as `(H, W)`, and both now come back as `(2, H, W)`. Mask lists, unbatched images, batch-1 images and masks with one entry per image are unchanged. An `(H, W)` mask is now checked against the image's height and width like a 3-D or 4-D mask, and a mismatch raises the same `ValueError`; it used to go through unchecked (a `(9, 8)` mask with `(8, 8)` images came back with height 9). ([#5618](https://github.com/kornia/kornia/pull/5618))
- `AugmentationSequential.inverse` now reads a 3-D mask the way the forward pass does: a `(B, H, W)` mask whose
  leading size is the batch size is inverted per sample, and under `keepdim=True` (on the container or the child) the
  inverse returns the rank it was given. Previously the inverse read every 3-D mask as one `(C, H, W)` image:
  per-sample warps and resample-mode crops (`RandomAffine`, `RandomPerspective`, `RandomCrop`, ...) raised a
  batch-size error, `RandomRotation` returned `(B, B, H, W)`, and flips and `Resize` returned `(1, B, H, W)`. Under
  `keepdim=True`, `(H, W)`, `(1, H, W)` and `(C, H, W)` masks used to come back as `(1, 1, H, W)` or `(1, C, H, W)`
  and now keep their shape; under `keepdim=False`, a `(B, H, W)` mask passed to `inverse` now comes back as
  `(B, 1, H, W)` instead of `(1, B, H, W)`. ([#5619](https://github.com/kornia/kornia/pull/5619))
- On an empty batch, the inverse of a shape-changing 2D geometric augmentation now returns the input height and width
  recorded in its parameters, keeping the channels, dtype, device and autograd graph of the tensor it inverts. It used
  to return the transformed size: for a `(0, 3, 6, 8)` forward, `Resize((5, 7)).inverse` gave `(0, 3, 5, 7)` and now
  gives `(0, 3, 6, 8)`. This covers `Resize`, `LongestMaxSize`, `SmallestMaxSize`, `CenterCrop`, `RandomResizedCrop`
  and resample-mode `RandomCrop`, directly and through `AugmentationSequential.inverse`. Two empty-batch parameters
  change with it. `RandomCrop` now records the unpadded input as `forward_input_shape`; it used to record the padded
  canvas, which an empty batch cannot undo because its `padding_size` has no rows. In a container, a stage after a
  `RandomCrop` or `Resize` (including `LongestMaxSize` and `SmallestMaxSize`), whatever its `p`, now records its
  own input size; it used to record the size before that crop or resize. Empty `RandomCrop` parameters saved before
  this change still replay but restore the padded canvas, for example `(0, 1, 12, 12)` instead of `(0, 1, 6, 8)` with
  `padding=(1, 2, 3, 4)`: regenerate them. Non-empty batches are unchanged. Part of #4429. ([#5620](https://github.com/kornia/kornia/pull/5620))
- `AugmentationSequential` now repeats a single mask to the batch in two more cases, so each sample's own parameters apply to its copy and the mask comes back with batch `B`: a `(1, T, C, H, W)` mask next to a `(B, T, C, H, W)` video (a `VideoSequential` child; next to a BTCHW video a `(1, T, H, W)` mask too), and a mask replayed alone with `params=` and no image, which is repeated to the batch recorded in those params (`forward_input_shape[0]`; for a video, a new `video_batch_size` entry that `VideoSequential` now adds to each child's parameters). Previously both came back at batch 1: flips and crops applied sample 0's parameters only, and `RandomAffine` raised `grid_sampler(): expected grid and input to have same batch size`. For example, with a batch-2 video a `(1, T, 1, H, W)` mask through `RandomHorizontalFlip` used to come back as `(1, T, 1, H, W)` and now comes back as `(2, T, 1, H, W)`. Some inputs that used to run, or to fail late, now raise `ValueError`: a mask next to a video that is neither 5D nor a 4D mask next to a BTCHW video (it failed later inside the `VideoSequential` reshape or came back with a wrong shape; with `same_on_frame=False` a `(1, 4, 8, 8)` mask next to a 4-frame BCTHW video came back as `(1, 1, 8, 4, 8)`), a 1D mask (it used to raise `RuntimeError`), a 5D mask next to a non-video image, a mask whose frame count differs from the video's, and a replayed mask whose batch is neither 1 nor the recorded batch (`RandomHorizontalFlip` and `RandomCrop` used to run silently with a wrong result). Full-batch masks and 3D-augmentation containers are unchanged. A `VideoSequential` nested inside another container keeps the old behavior. ([#5636](https://github.com/kornia/kornia/pull/5636))
- `mean_average_precision` now raises kornia's `BaseError`, naming `pred_labels` or `gt_labels`, when a floating-point
  label is not a whole number (#5629). Previously a fractional label such as `1.5` passed the range check and matched no
  class, so the call returned a value without an error: a detection with such a label was dropped, and an object with
  such a label was not counted (with two perfect detections labelled `[1.5, 2.0]`, class 1 read `-1.0` and the mAP was
  the AP of class 2 alone). Integer, `bool` and integer-valued floating-point labels such as `1.0` give the same results
  as before. ([#5638](https://github.com/kornia/kornia/pull/5638))
- In a container, a stage after an always-applied (`p=1`) `RandomResizedCrop` on an empty batch now records the crop's
  output size as its `forward_input_shape`, as for a non-empty batch; it used to record the size before the crop. For a
  `(0, 3, 9, 13)` batch through `RandomResizedCrop((4, 5))` and then `Resize((2, 3))`, the `Resize` stage used to
  record `(0, 3, 9, 13)`, so inverting that stage alone returned `(0, 3, 9, 13)`; it now records `(0, 3, 4, 5)` and
  its inverse returns `(0, 3, 4, 5)`. The inverse of the whole container still returns `(0, 3, 9, 13)`. Empty
  parameter lists saved before this change keep the old sizes: regenerate them. Non-empty batches are unchanged.
  Part of #4429. ([#5647](https://github.com/kornia/kornia/pull/5647))
- `Boxes.filter_boxes_by_area` now marks the boxes it removes as invalid instead of only zeroing their stored inclusive vertices. Previously a removed box exported as `[0, 0, 1, 1]` in the exclusive `'xyxy'`, `'xywh'` and `'vertices'` modes, `to_mask` filled pixel `(0, 0)` for it, and a later `transform_boxes`, `translate` or `pad` moved it like a real box. Now it exports as exact zeros in every mode, also after later transforms, and `to_mask` draws nothing for it. `RandomMosaic` boxes below `min_bbox_size` change in the same way: in direct calls for every box key, and inside `AugmentationSequential` for `"bbox_xywh"`, and for every key when a geometric augmentation follows the mosaic (it used to move them into real-looking boxes; without one, `"bbox_xyxy"` and `"bbox"` already came out as zeros). Because zero-width rows are rejected on import, passing such an `'xyxy'` or `'xywh'` export back to `Boxes.from_tensor` with the default `validate_boxes=True`, or a `"bbox_xywh"` output to another `AugmentationSequential`, now raises `ValueError` where it used to import a 1x1 box at the origin. A `Boxes` object pickled by an earlier version loads with every box valid. ([#5649](https://github.com/kornia/kornia/pull/5649))
- `equalize_clahe` and `RandomClahe` now spread the residual of the clipped pixels the way OpenCV does, one count on every `max(256 // residual, 1)`-th histogram bin from bin 0, in the fast, per-sample `clip_limit` and differentiable paths. They used to write the residual into the first `residual` bins, which lifted every tile's lookup table: wherever clipping left a residual, the output was brighter than `cv2.createCLAHE`, by about 40 gray levels on average for 64 x 80 to 96 x 96 images with an 8 x 8 grid and clip limit 2.0, and by 4 to 5 levels at 224 x 224. The output now changes for every tile whose clipping leaves a residual (#5654). On the images we measured (64 x 80 to 512 x 512, 8 x 8 grid, clip limits 2, 4 and 40), the mean signed difference from OpenCV is now within 0.1 gray levels in the fast, per-sample and differentiable paths, with the fast paths rounding each lookup-table entry as OpenCV does (#5659). ([#5655](https://github.com/kornia/kornia/pull/5655))
- For an integer or bool input, `filter2d`, `filter2d_separable`, `filter3d`, `gaussian_blur2d`, `box_blur`,
  `laplacian`, `motion_blur`, `pyrdown` and `pyrup`, the modules built on them (`GaussianBlur2d`, `BoxBlur`, `Laplacian`,
  `MotionBlur`, `PyrDown`, `PyrUp`), the `ScalePyramid` levels, every `build_pyramid` level after the input itself, and
  the edge map of `canny` / `Canny` now return float32 on the input's scale, without rounding or clamping. They used to
  return the input's integer dtype, holding zeros for a fractional kernel and wrapped sums for an integer one.
  `joint_bilateral_blur` with an integer `input` and a float64 `guidance` now returns float32 instead of float64.
  Floating inputs keep their dtype and values (#5155). ([#5669](https://github.com/kornia/kornia/pull/5669))
- `equalize_clahe` and `RandomClahe` now round fast-path lookup-table entries to the nearest integer (ties to even), matching OpenCV's LUT quantization, instead of flooring them. In float32 and float64 each fast-path output pixel rises by at most one gray level (1/255), and for the same tile its lookup table now equals OpenCV's (#5659). The differentiable path is unchanged. ([#5679](https://github.com/kornia/kornia/pull/5679))
- `TensorWrapper`, `Vector2`, `Vector3` and `Scalar` results now get one type rule, whether a torch function, a tensor method, a tensor attribute or an operator produced them. A result takes the class of a `Vector2`/`Vector3` operand, on either side, when it ends in that width and has at least as many axes as that operand. Otherwise it takes the class of the first operand without a width, and otherwise it is a plain tensor. Before, a tensor method such as `v.clone()` returned a plain tensor while `torch.clone(v)` returned a `Vector3`, `Scalar * Vector3` returned a `Scalar` holding the vectors, and `torch.linalg.norm(v, dim=-1)` and `v.T` raised unless the result happened to end in 3, in which case it was returned as a `Vector3`. Now `v.clone()` and `Scalar * Vector3` are a `Vector3`, `torch.linalg.norm(v, dim=-1)` returns a plain tensor, and `v.T` returns a plain tensor unless the batch holds exactly 3 vectors, whose `(3, 3)` transpose stays a `Vector3` because the rule reads shapes only. The same applies to any torch function that removes an axis: `torch.mean(v, dim=0)` of an `(N, 3)` `Vector3` was a `Vector3` and is now a plain `(3,)` tensor, as `v.mean(dim=0)` already was. `Vector2`/`Vector3` indexing now uses `data[key]`: `v[..., 0]` and `v[:, 0]`, which raised, and an unbatched `v[0]`, which raised `IndexError`, return the plain tensor the wrapped tensor gives, and keys that keep the coordinate axis return the vector class as before. ([#5688](https://github.com/kornia/kornia/pull/5688))
- A tuple `forward` output of `ImageModule`, `ImageSequential` and the augmentation containers used
  to come back as a list, and a one-element tuple as its bare element. It now keeps the tuple type
  and its length. A caller that used the bare element of a one-element tuple output must index it:
  `out = module(x)[0]`. A caller that relied on the list type of a multi-element tuple output
  (`isinstance(out, list)`, `out.append`) must convert with `list(out)`; one that indexed it
  (`out[0]`) needs no change. ([#5695](https://github.com/kornia/kornia/pull/5695))
- On an empty batch, a shape-changing 2D augmentation now gives a mask the height and width it
  gives the image. It used to return the mask at its input size: `AugmentationSequential(Resize((4, 5)),
  data_keys=["input", "mask"])` on a `(0, 3, 8, 10)` image and a `(0, 1, 8, 10)` mask gave a `(0, 3, 4, 5)` image next
  to a `(0, 1, 8, 10)` mask, and passing that pair to `AugmentationSequential.inverse` raised `ValueError`. The mask is
  now `(0, 1, 4, 5)` and the inverse restores `(0, 1, 8, 10)`. This covers `Resize`, `LongestMaxSize`,
  `SmallestMaxSize`, `CenterCrop`, `RandomCrop`, `RandomResizedCrop` and `PadTo`, in `AugmentationSequential` (also with
  dictionary inputs and inside `VideoSequential`) and in a direct `transform_masks` call. This holds
  for any `p`: an empty image and mask both take the configured size. Non-empty batches are unchanged. Part of #4429. ([#5706](https://github.com/kornia/kornia/pull/5706))
- `AugmentationSequential` processed a mask in its image's dtype, so integer labels above that dtype's exact range were silently rounded: `2049` became `2048` next to a float16 image. It now processes integer and boolean masks in at least float32 (float64 next to a float64 image), so their labels come back intact. A label that is not exactly representable even there (for example `2**24 + 1` next to a float32 or half-precision image) now raises `RuntimeError` instead of being rounded; remap the labels to smaller integers or use a float64 image where supported. Next to a float16 or bfloat16 image, a warped integer or boolean mask (rotation, affine, perspective, ...) is now resampled on a float32 grid, so it can differ from before at individual pixels; the half-precision resampling on CPU also picked wrong source pixels, including labels the mask does not contain. Floating masks keep the image's working dtype. ([#5711](https://github.com/kornia/kornia/pull/5711))
- `Affine` now uses `shear` like `angle`, `translation` and `scale_factor` when it creates the parameters it was not
  given and when it checks that its parameters share one device and dtype. An `Affine` given only `shear` now creates its
  default rotation, translation and scale on the shear's device and in its dtype, so it works on float64, float16 and
  bfloat16 inputs and on accelerators. Before, those defaults were float32 tensors on the default device, and the forward
  raised `ValueError` on any other input. A `shear` on another device or of another dtype than the other parameters now
  raises `DeviceError` or `TypeCheckError` when the module is built, as a mismatch among `angle`, `translation` and
  `scale_factor` already did. Before, such a module was built and its forward raised, except for a `shear` of lower
  precision than parameters matching the input (a float32 `shear` with a float64 `angle` on a float64 image), which
  worked. An integer `shear` given alone now gets integer defaults, and its forward raises `ValueError`, as it already
  did for an integer `angle`, `translation` or `scale_factor` given alone; before, it was combined with float32 defaults. ([#5715](https://github.com/kornia/kornia/pull/5715))
- `equalize_clahe` and `RandomClahe` now blend neighbouring tile lookup tables with weight steps of `1 / tile_size`, instead of `1 / (tile_size - 1)`, anchored at each tile's actual centre. This changes interpolated output values in both the fast and differentiable paths. Kornia intentionally anchors the blend half a pixel before OpenCV/scikit-image's historical phase to preserve flip symmetry on unpadded grids. For the same tiles and lookup tables, the phase difference is at most half an interpolation step per axis; exact `cv2.createCLAHE` parity is not the contract. On the images we measured with an 8 x 8 grid and even tiles (64 x 80 to 512 x 512, clip limits 2 and 40), the largest difference from `cv2.createCLAHE` fell to between one half and two thirds of what it was (21 instead of 39 gray levels on a smooth 64 x 80 image at clip limit 40, 3 instead of 5 on the 512 x 512 `camera` image), while the mean difference stayed close to what it was (7.0 instead of 7.6 and 0.54 instead of 0.57 levels), because the half-pixel offset from OpenCV's phase changes every blended pixel. Border regions still clamp to the nearest tile along the border axis. LUT calculation and quantization are unchanged. ([#5716](https://github.com/kornia/kornia/pull/5716))
- `Hyperplane` now registers its normal and offset as module state under the keys `_n` and `_d`: `state_dict()`
  saves them, `load_state_dict()` restores them, and `.to()`, `.double()` and an enclosing module's conversions move
  and cast them. Previously a plane's `state_dict()` was empty, loading a state dict into a module holding a plane
  reported success and kept the old plane, and `.to()` left the plane on its original device and dtype.
  As in `ParametrizedLine`, a floating-point tensor without autograd history is now stored as an `nn.Parameter`. So a
  plane built from plain tensors, including the result of `Hyperplane.through` or `fit_plane` on points without autograd
  history, has two trainable parameters (an optimizer built from `model.parameters()` now updates them), and the distances
  and projections it returns require grad (call `.detach()` before `.numpy()`). Previously such a plane had no parameters
  and its outputs did not require grad. A caller's `nn.Parameter` stays a parameter, a tensor with autograd history keeps
  it, and an integer tensor is stored as a buffer.
  A state dict saved before this change has no plane keys, so loading it with `strict=True` now raises on the missing
  `_n` and `_d` keys; `strict=False` reports them and keeps the plane the module was built with, as before. A whole
  module saved with `torch.save` before this change still loads with its plane, which is moved to registered state. ([#5717](https://github.com/kornia/kornia/pull/5717))
- `DenseSIFTDescriptor` output is smaller at the default `padding=1`. It used to return an `H x W` grid, one descriptor per
  pixel, built from cells one pixel apart. It now returns `(H - 9) x (W - 9)`, with each output centred at input position
  `j + 4.5` along each axis, and an input under 10 pixels per side raises. In general the height is
  `(H + e + 2 * padding - (num_spatial_bins - 1) * spatial_bin_size - 1) // stride + 1`, where `e` is 1 for an even
  `spatial_bin_size` and 0 otherwise; pass `padding=6` for `H + 1` at the other defaults. ([#5734](https://github.com/kornia/kornia/pull/5734))
- `make_grid`, `ImageModule.show` and `ImageModule.save` now read `n_row` as the number of images per row, as their
  docstrings say (the meaning of `nrow` in `torchvision.utils.make_grid`). Previously `n_row` set the number of rows, so
  every grid whose row and column counts differ was laid out the other way round: 6 images with `n_row=2` used to give
  2 rows of 3 and now give 3 rows of 2. The default `n_row=None` (the square root of the batch size, rounded up) is now
  the number of columns, so 2 images used to be stacked vertically and are now side by side, and 5 or 6 images used to
  be 3 rows of 2 and are now 2 rows of 3. Images still fill the grid row by row in batch order. To keep the old
  placement of `B` images in `R` rows, pass `n_row=math.ceil(B / R)`. (#5745) ([#5745](https://github.com/kornia/kornia/pull/5745))
- The kernel builders raise for an integer or bool `dtype` that cannot hold their taps, where they used to return a
  wrong kernel. `get_box_kernel1d`, `get_box_kernel2d`, `get_gaussian_kernel1d`, `get_gaussian_erf_kernel1d`,
  `get_gaussian_discrete_kernel1d`, `get_gaussian_kernel2d`, `get_gaussian_kernel3d`, `gaussian`, `gaussian_discrete`,
  `gaussian_discrete_erf`, `get_diff_kernel3d`, `get_diff_kernel3d_2nd_order` and `get_spatial_gradient_kernel3d` have
  fractional taps and now raise for every integer and bool `dtype` (for the Gaussians, the `dtype` that goes with a
  Python `sigma`). They used to return:

  - for the box kernels, the kernel in that dtype with every tap truncated to 0 (all `True` in bool), except a size-1
    box, which was all ones and now raises too;
  - for the sampled and erf Gaussians, a floating kernel for `sigma` truncated to an integer, so
    `get_gaussian_kernel1d(5, 1.5, dtype=torch.int32)` was the `sigma=1` kernel and an integer `sigma` such as `2` gave
    the right kernel (it now raises); in uint8 the offsets also wrapped, so the taps before the centre were wrong (0 for
    a `sigma` below about 15);
  - for the discrete Gaussians, the kernel in that dtype with every tap truncated to 0;
  - for the 3-D difference kernels in a signed dtype, the kernel with the half and quarter taps truncated to 0, so the
    first-order channels and the mixed second-order ones were all zero (bool gave `True`/`False` taps).

  A bool `dtype` already raised for the Gaussians, and uint8 already raised for the 3-D difference kernels.
  `get_laplacian_kernel1d`, `get_laplacian_kernel2d` and `laplacian_1d` keep accepting a signed integer `dtype` wide
  enough for their centre tap and raise for uint8, bool and a narrower one, where they used to wrap the centre (to 252
  for a uint8 size 5, to 88 for an int8 13x13) or make every tap `True`. `spatial_gradient3d` and `SpatialGradient3d`
  now return float32 for an int8, int16, int32 or int64 input with `order=2`; they used to return the input's dtype,
  with the mixed second derivatives zero.

  ([#5759](https://github.com/kornia/kornia/pull/5759))
- `RandomAffine` and `RandomShear` now check a `shear` given as a 2x2 tensor (or nested pairs
  `((x_min, x_max), (y_min, y_max))`) the way they check the flat `(x_min, x_max, y_min, y_max)` form: each
  axis range must be finite, ordered and inside `[-360, 360]`, otherwise the constructor raises `ValueError`
  (`shear-x out of bounds. Expected inside (-360, 360)`, `If shear-y is a range, it must be finite`).
  Previously the 2x2 form skipped the bound and finiteness checks: an endpoint past 360 degrees was
  accepted, and an infinite endpoint was accepted and produced NaN images at forward time. A reversed range
  was already rejected; its message now names the axis (`shear-x[0] should be smaller than shear-x[1]`). ([#5766](https://github.com/kornia/kornia/pull/5766))
- `build_pyramid` and `build_laplacian_pyramid` now reject non-positive integer `max_level` values with a named
  validation error instead of returning a single level. Non-integer values, including booleans previously treated as
  integers, are rejected consistently. `ImageRegistrator.register` builds its pyramids with `build_pyramid`, so
  `pyramid_levels <= 0` (or a boolean) now raises the same error instead of registering at the full resolution only.
  Positive integer levels and the existing padding behavior are unchanged. ([#5778](https://github.com/kornia/kornia/pull/5778))
- `AugmentationSequential` mask lists now hold one tensor per image sample, each `(H, W)`, `(C, H, W)` or `(1, C, H, W)`, and every entry follows its own sample's gate, parameters and transformation matrix in forward, replay, nested sequences, auto-augment policies and inverse. Before, every entry received the whole batch's parameters with only sample `i`'s gate: per-sample entries failed in warps or were cropped with sample 0's window, full-batch entries were transformed whole but gated by one sample's draw, and a list longer than the batch raised `IndexError`.
  A list of full-batch tensors, a list whose length differs from the batch, or a non-tensor entry now raises `ValueError`, with or without an image; pass full-batch masks under separate `mask` data keys. Video, mix and 3D children and a geometric `PatchSequential` raise `NotImplementedError` for a mask list; before, mix children returned the list unchanged while the image was mixed. Mask-only replay and inverse retain every sample (before, they returned a single tensor or raised `IndexError`), and list calls preserve the child's full-batch parameters.
  A tensor mask next to a rank-2 `(H, W)` image is now checked as next to a `(C, H, W)` image: another height or width, a 4D batch above 1 or a 5D mask raises `ValueError`. Before, it was transformed as given, cropped with the image's window, or failed inside a child with `IndexError` or `RuntimeError`. ([#5779](https://github.com/kornia/kornia/pull/5779))
- `CenterCrop3D` and `RandomCrop3D` now name `size` and the value passed in their size errors. A `size` of the wrong type or length, or with a float element, raises `TypeError` in the constructor: before, `CenterCrop3D((2, 2))` raised `IndexError`, `CenterCrop3D((2, 2, 2, 2))` silently cropped to `(2, 2, 2)`, `RandomCrop3D` raised `AssertionError` at the first call for an `int` or a sequence of other than three elements, and a float element raised `TypeError` from `affine_grid` at the first call in both classes. A crop larger than the input along any axis now raises `ValueError` in both classes, with the requested and input spatial sizes; before, `CenterCrop3D` raised `AssertionError`. `CenterCrop3D(n)` still works and `RandomCrop3D(n)` still does not: pass `(n, n, n)`. ([#5789](https://github.com/kornia/kornia/pull/5789))
- `kornia.contrib.histogram_matching` no longer extrapolates below the template minimum. When the template's darkest value covers a larger fraction of its pixels than the source's darkest value does (for example a template with clipped blacks), the source pixels whose quantile is below that fraction used to map below the template minimum, to negative intensities for a template whose minimum is 0; they now map to the template minimum, as in `skimage.exposure.match_histograms`. A constant template used to return NaN everywhere and now returns that constant. ([#5794](https://github.com/kornia/kornia/pull/5794))
- The kornia_rs floor is 0.2.0, the release a plain `pip install kornia` resolves and the one CI tests;
  kornia_rs 0.1.x is no longer tested. ([#5797](https://github.com/kornia/kornia/pull/5797))
- `kornia.onnx.ONNXSequential` and `kornia.onnx.ONNXModule` raise a `ValueError` naming the expected and received counts when they are called with a different number of arrays than the session has inputs. Extra arrays used to be dropped without a warning, so the call returned the result for the leading arrays only, and too few arrays raised an `IndexError`. ([#5799](https://github.com/kornia/kornia/pull/5799))
- `resize_to_be_divisible` rounds a side of at most half of `divisible_factor` up to `divisible_factor` instead of down to 0: a `(1, 3, 10, 14)` image with factor 24 now gives `(1, 3, 24, 24)` where it gave an empty `(1, 3, 0, 24)` image. An empty side still stays empty. It reads height and width from the last two axes, so `(H, W)` inputs and inputs with more than one leading axis are resized like in `resize`; they used to raise `UnboundLocalError`. A `divisible_factor` that is not a positive integer raises `ValueError`: a negative factor used to give the size of its absolute value, 0 raised `ZeroDivisionError` and a float raised `TypeError`. ([#5802](https://github.com/kornia/kornia/pull/5802))
- `compute_padding` now pads an axis that is shorter than the window up to one full window. It used to return only the stride-alignment padding, which can leave the padded image smaller than the window: for a 4x6 image with an 8x8 window and stride 2 it returned `(0, 0, 0, 0)`, and `extract_tensor_patches(..., allow_auto_padding=True)` and `combine_tensor_patches(..., allow_auto_unpadding=True)` raised a `RuntimeError`. They now extract one window along such an axis and reconstruct the image. Results are unchanged wherever the old padding already fitted a window. ([#5806](https://github.com/kornia/kornia/pull/5806))
- `load_image(path, ImageLoadType.UNCHANGED)` returns a grayscale JPEG as one channel, `(1, H, W)`; it used to return
  three equal channels. Image modules given a file path, such as `AugmentationSequential` and `ImageSequential`, now
  receive `(1, 1, H, W)` for such a file instead of `(1, 3, H, W)`, so a colour augmentation such as `ColorJiggle`
  raises for it, as it already did for a grayscale PNG path. Invalid arguments raise a typed kornia error up front:
  `load_image` raises `TypeCheckError` (a `TypeError`) when `desired_type` is not an `ImageLoadType`, where it raised
  `NotImplementedError`, or `AttributeError` for a 16-bit file. `write_image` raises `TypeCheckError` for an input that
  is not a tensor, where it raised `AttributeError`. It validates `quality` for every format, including PNG and TIFF,
  which used to ignore it: a value that is not an integer raises `TypeCheckError`, and one outside [0, 100] raises
  `ValueCheckError` (a `ValueError`), where a JPEG accepted 101 to 255 (101 gave the same file as 100) and raised
  `OverflowError` beyond that. A batched `(B, C, H, W)` tensor raises `ShapeError` (a `ValueError`) naming the shape,
  where kornia-rs raised a `TypeError`. Writing into a directory that does not exist raises `FileNotFoundError` naming
  the directory, and writing under a path that is a file raises `NotADirectoryError`; both used to raise a bare
  `Exception`. `get_sample_images` caches downloads under `<kornia_config.hub_cache_dir>/images` by default instead of
  a hard-coded `.kornia_hub/images`; the two are the same path unless `hub_cache_dir` is changed. A download that is not
  an image raises `ValueError` and is not cached, where it raised `PIL.UnidentifiedImageError`, an `OSError`, so an
  `except OSError` clause no longer catches it. ([#5808](https://github.com/kornia/kornia/pull/5808))
- `draw_point2d`, `draw_line` and `draw_rectangle` place a fractional coordinate on the pixel whose centre is nearest,
  a coordinate exactly halfway rounding up; they used to truncate toward zero, so 2.7 drew at pixel 2 and now draws at
  pixel 3. `draw_point2d` and `draw_line` check their bounds after rounding, so a coordinate lies inside the image
  from -0.5 up to, but not including, `W - 0.5` (`H - 0.5` for y). From `W - 0.5` up to `W`, both used to draw in the
  last column and now raise `ValueError`. `draw_point2d` also raises `ValueError` for any other point outside the
  image: a coordinate between -1 and -0.5 used to draw in the first column, one of -1 or less at the opposite edge,
  and one of `W` or more raised `IndexError`. `draw_line` accepts -0.5 up to 0, which used to raise.

  `draw_rectangle` no longer draws a side that lies outside the image, and it clips a fill to the image. It used to
  clamp the corners, which drew such sides along the border; clamp the corners to `[0, W - 1]` and `[0, H - 1]` for
  the old look. It accepts the two corners in either order, where reversed corners drew nothing. It raises
  `ValueError` instead of `AssertionError` when the image and rectangle batch sizes differ or the last dimension is
  not 4, and `ShapeError`, still a `ValueError`, naming the expected shape for an image that is not `(B, C, H, W)`.

  `draw_line` draws, at each step along the longer axis, the pixel nearest to the segment, so a line is the same from
  `p1` to `p2` as from `p2` to `p1`. It used to round the shorter-axis coordinate up from `p1`, which put sloped lines
  up to one pixel off and depended on the direction.

  `draw_point2d` raises `ValueError` for an image that is not `(C, H, W)` or `(H, W)`; a batched image used to be
  drawn along the wrong axes or raise a torch indexing error.

  `Image.to` and `Image.float` return a new `Image` and leave the original unchanged, like `torch.Tensor.to`; they
  used to change the image in place and return it, so code that ignores the result must now assign it
  (`img = img.float()`). `Image` checks its pixel format: `GRAY` needs one channel and `RGB` and `BGR` need three
  (`UNKNOWN` takes any count), and `bit_depth` must be the storage size, `8 * data.element_size()`, where `bit_depth`
  9 to 15 used to pass on `uint8` data and 12 failed on `uint16`. A mismatch raises `ValueCheckError`, a `ValueError`.
  Without a colour space, `Image.from_numpy` and `Image.from_dlpack` label one channel `GRAY`, three `RGB` and any
  other count `UNKNOWN`; they used to label every image `RGB`.

  `image_to_string` and `print_image` show each pixel with the nearest of the 240 fixed xterm colours (codes 16 to
  255, the gray ramp included), so printed colours change; they used to pick the nearest of the six colour-cube levels
  in each channel after truncating a float value to a whole level. `print_image` raises `TypeError` instead of
  `RuntimeError` for an input that is neither a path nor a tensor.

  `image_to_tensor` raises `TypeError` for any input that is not a `numpy.ndarray`, where a list raised
  `AttributeError`. `make_grid` raises `ValueError` for a negative `padding`, which used to crop the images.

  `draw_point2d` no longer exports with `torch.export`, nor to ONNX from torch 2.9, because its out-of-image check
  reads tensor values. `draw_rectangle` exports with `torch.export` from torch 2.14, which it did not before. ([#5813](https://github.com/kornia/kornia/pull/5813))
- `pyrdown` and `PyrDown` now compute the output width with the same rule as the height, `int(width / factor)`, so both axes follow the documented `floor(side / factor)`. The width used floor division of the floating-point values, which could drop a pixel when `width / factor` is a whole number and the factor's binary value lies slightly above its decimal value (1.1, 1.3, 1.6, 1.8, 2.2, ...): at `factor=1.6` a 480 x 640 image used to give 300 x 399 and now gives 300 x 400. Every width that is a multiple of 8 at `factor=1.6`, of 13 at 1.3 or of 9 at 1.8, and some multiples of 11 at 1.1, gains that pixel, and the output values change with the new sampling grid. Integer and dyadic factors (2.0, 3.0, 1.5), KeyNet's internal 1.2 and the default `scale_factor_levels` of `MultiResolutionDetector` (`sqrt(2)`) give the same sizes as before; a `MultiResolutionDetector` configured with an affected `scale_factor_levels` can return different detections. ([#5817](https://github.com/kornia/kornia/pull/5817))
- `ImageModuleMixIn`, `ImageSequential` and `AugmentationSequential` now load an 8-bit palette PNG image path as its RGB
  colors, or RGBA colors when the file has a `tRNS` chunk, as they already did for the same image passed as a PIL object.
  Previously such a path produced one channel of palette indices divided by 255. A palette PNG path whose palette is too
  short for its pixels, or whose `tRNS` chunk fails its CRC check, used to load as indices and now raises `ValueError`,
  as `kornia.io.load_image` does for that file under every type except `UNCHANGED`. ([#5833](https://github.com/kornia/kornia/pull/5833))
- `equalize_clahe` and `RandomClahe` now use tiles of ceil(H / grid_size[0]) by ceil(W / grid_size[1]) pixels, odd or even, and reflect-pad only as needed to complete the grid. Previously, odd tile sizes were rounded up to the next even size and the image was padded to fit, even when the grid divided it. For example, an 8 x 8 grid on a 72 x 88 image used 10 x 12 tiles over an 80 x 96 padded image, shifting tile boundaries and changing output values. These are OpenCV's tiles when the grid divides both image axes or neither. When it divides only one, OpenCV also pads that axis, by one pixel per tile, and Kornia does not, so on such images the difference from `cv2.createCLAHE` can grow. On the smooth images we measured with an 8 x 8 grid at clip limit 2, the mean / max difference from `cv2.createCLAHE` fell from 8.0 / 47 to 3.2 / 10 gray levels at 72 x 88 and from 18.8 / 82 to 1.0 / 3 at 100 x 150, and odd tiles on a grid that divides the image are now flip-symmetric. ([#5845](https://github.com/kornia/kornia/pull/5845))
- Removed the legacy 12-vector pinhole helpers `pinhole_matrix`, `inverse_pinhole_matrix`, `scale_pinhole`, `get_optical_pose_base` and `homography_i_H_ref` from `kornia.geometry.camera.pinhole`. They were never exported or documented, and their docstrings had announced their deprecation in favour of `PinholeCamera` since before kornia 0.2. `pinhole_matrix` added `eps` to every entry of its output, `inverse_pinhole_matrix` inverted a perturbed matrix, and `get_optical_pose_base` and `homography_i_H_ref` always raised `NotImplementedError`. Importing them from the module now raises `ImportError`. Use `PinholeCamera` instead: `intrinsics` for the 4×4 matrix returned by `pinhole_matrix`, `intrinsics_inverse()` for its 4×4 inverse (`camera_matrix` returns only the 3×3 intrinsics block), `scale()` to rescale, and `extrinsics` for the pose (#4268, #4283). ([#5874](https://github.com/kornia/kornia/pull/5874))
- `ALIKED` and `DISK` no longer pad their keypoints with non-maxima, nor drop a real maximum, when an image has no more
  local maxima than the keypoint budget. In top-k mode (`ALIKED(max_num_keypoints=k, detection_threshold<=0)`), ALIKED
  used to return exactly `k` keypoints per image, filling the slots beyond the score map's maxima with other positions
  in `torch.topk`'s tie order, including the removed border, the region a `forward_laf` `mask` excludes and the
  `image_size` padding, and it raised when `k` exceeded the number of pixels. It now returns at most `k`, all of them
  maxima, and `forward_laf` pads each image to the largest count of the batch with zero frames (#5862).
  `DISK(...)(images, n=n)` used to drop the weakest maximum of an image with `n` maxima above `score_threshold` or fewer,
  and raised on an image with none; it now returns all of them, the keypoints `n=None` returns. With more than `n`, it
  used to leave out every maximum tied across the `n`-th place, returning fewer than `n`; it now keeps the `n` strongest,
  taking maxima tied at the `n`-th score in raster order (#5864). In threshold mode, ALIKED fell back to
  the maxima above the mean score only when no image of the batch had a maximum above `detection_threshold`; the
  fallback is now decided for each image, so a batch returns the keypoints that one-by-one calls return (#5865). The
  default `ALIKED` configuration on one image at a time, and `DISK` on an image with more than `n` maxima and no tie across
  the `n`-th place, return the same keypoints as before. ([#5875](https://github.com/kornia/kornia/pull/5875))
- `kornia.contrib.histogram_matching` returns the source dtype for a floating source and `float32` for an
  integer source, whatever the template dtype. The output dtype used to depend on the template too: a
  `float32` source with a `float64` template returned `float64` (now `float32`), a `float16` source with a
  `float32` template returned `float32` (now `float16`), and a `uint8` source with a `float64` template
  returned `float64` (now `float32`). Pairs of the same dtype return the same dtype as before. ([#5878](https://github.com/kornia/kornia/pull/5878))
- `VideoSequential.inverse_keypoints` called with a raw `(B, T, N, 2)` tensor used to return it shaped `(T, B, N, 2)`, with the values still in `(B, T)` order; it now returns `(B, T, N, 2)`, as `transform_keypoints` does. The raw-tensor branches of `transform_keypoints`, `inverse_keypoints`, `transform_boxes` and `inverse_boxes` used to raise for zero keypoints or boxes, an empty video batch and a non-contiguous tensor; they now return the input's shape. ([#5881](https://github.com/kornia/kornia/pull/5881))
- `kornia.core.utils.dict_to_dataclass` now raises `TypeCheckError` for a value that is neither a dict nor an instance
  of the dataclass its annotation names, wherever the annotation names it: a list or tuple element, a dict value, or a
  field whose annotation is a string, which includes every field of a module with `from __future__ import annotations`.
  Such values used to pass through unchanged; a field annotated with the dataclass class itself already raised. For
  example, `[5]` for a `List[Inner]` field, `["x", 2]` for a `Tuple[Inner, int]` field, `{"k": 5}` for a
  `Dict[str, Inner]` field and `None` for a `"Inner"` field used to be stored as given and now raise, and in a module
  with `from __future__ import annotations` the round trip of a `sub: Sub = None` default through `dataclass_to_dict`
  now raises: annotate the field `Optional[Sub]`.

  `ParametrizedLine`, `So2`, `Se2` and `Se3` given an integer or bool tensor now raise `TypeCheckError` at construction,
  naming the field and its dtype; cast the tensor to a floating-point dtype first. In eager mode they used to raise torch's
  "Only Tensors of floating point and complex dtype can require gradients" `RuntimeError`. Under `torch.compile`,
  `torch.jit.trace` and export they used to keep the integer tensor as a plain attribute, and now raise as well.
  `Hyperplane` still stores integer coefficients as buffers. ([#5910](https://github.com/kornia/kornia/pull/5910))
- `kornia.onnx.download.CachedDownloader.download_to_cache` takes `cache_dir` and `suffix` as keyword-only parameters and raises `TypeError` for any other keyword. `ONNXLoader.load_model` and `ONNXLoader.load_config` raise it for any keyword but `cache_dir`, whatever the model name or URL, and `HFONNXComunnityModelLoader.load_model` for any keyword. An unknown keyword, such as a misspelt `cachedir=`, used to be ignored, and the file was cached in the default directory. `download_to_cache` also raises `ValueError` for a `name` that is absolute or has a drive or an empty, `.` or `..` component, and for a `suffix` that adds a directory to the file name; so does `ONNXLoader.load_model` for an `hf://` name whose folder is `..`, and `HFONNXComunnityModelLoader` for such a `model_name`. Such names used to be joined to the cache directory as they were: `..` stored the file outside the cache directory, and an absolute name was stored below it on POSIX (an absolute `model_name` replaced it). `ONNXLoader.load_model` and `load_config` raise `ValueError` for a URL without a host, such as `https:///a/m.onnx`, which `urlopen` refuses to fetch. ([#5911](https://github.com/kornia/kornia/pull/5911))
- `RandomMosaic` now checks `padding_mode` at construction, in both cropping modes, and raises a `ValueError` that names the value and lists `constant`, `reflect` and `replicate` for anything else. Previously the constructor accepted any value: `cropping_mode="slice"` ignored it, and `cropping_mode="resample"` passed any other string to `grid_sample`, so `grid_sample`'s own names `zeros`, `border` and `reflection` worked (as `constant`, `replicate` and `reflect`) while any other value failed inside `forward` with torch's `grid_sample` error. ([#5917](https://github.com/kornia/kornia/pull/5917))
- `So2`, `Se2` and `Se3` now reject at construction the inputs their methods cannot use. `So2(z)` with a real
  floating-point `z` used to construct, and `inverse()`, `adjoint()`, `__getitem__` and `So2 * So2` worked on it while
  `matrix()`, `log()` and `So2 * points` raised torch's "imag is not implemented"; it now raises `TypeCheckError`
  naming `z` and its dtype (build `z` with `torch.complex`). `Se2` / `Se3` with a translation whose dtype differs from
  the rotation's (for example a float64 rotation and a float32 translation) used to construct, and `matrix()`,
  `inverse()`, `__mul__` and `__getitem__` worked on some of them while `log()` (`Se2`) or `inverse()`, `adjoint()` and
  `Se3 * Se3` (`Se3`) raised torch's "expected scalar type" error; they now raise `TypeCheckError`. A complex `Se2` / `Se3` translation and an integer or bool `nn.Parameter` translation, which used to construct, also
  raise `TypeCheckError`. A translation on another device than the rotation, which made `matrix()`, `log()`, `inverse()`,
  `adjoint()` and `*` fail, now raises `DeviceError`. A bfloat16 `Se2` translation beside a complex32, complex64 or complex128 rotation stays accepted, because `.bfloat16()` leaves exactly
  that state, and every `Se2` method now works on it: `log()` used to raise torch's dtype error, and `inverse()` and
  `pose * pose` return a translation of the rotation's real dtype (float32 or float64 as before, but float16 instead of
  float32 for a complex32 rotation, which the new check would reject). ([#5925](https://github.com/kornia/kornia/pull/5925))
- `histogram_matching` with an empty template now raises a `ValueError` ("histogram_matching needs a non-empty template."); it used to raise an `IndexError` from inside the quantile computation. Passing `eps=` to `combine_tensor_patches` inside `torch.compile(fullgraph=True)` compiled before; `eps` now emits a `DeprecationWarning`, which makes that call fail with `Unsupported` under the default TorchDynamo configuration, so drop `eps` from the call. ([#5960](https://github.com/kornia/kornia/pull/5960))
- `connected_components` now returns `int64` labels with exact ids. It used to return the input dtype, which merged distinct components in `float16`, `bfloat16`, `bool` and `uint8` inputs (ids lost precision or wrapped) and, in any dtype, above 2**24 pixels. On MPS, which has no `float64`, an image of more than 2**24 pixels (a single image, not the batch) now raises `ValueError` instead of returning merged labels.

  `KMeans(tolerance=None, max_iterations=0)`, a negative `max_iterations` and a negative `tolerance` now raise `BaseError`. The first never stopped fitting. A negative `max_iterations` used to stop the fit after one iteration, and a negative `tolerance` used to be accepted and never triggered the early stop.

  `KMeans.predict` with a 1-D input now raises `ShapeError` instead of `IndexError`.

  `distance_transform` and `DistanceTransform` now reject an `h` so small that the kernel's corner tap `exp(-sqrt(ndim) * (kernel_size // 2) / h)` underflows, which the old axis-tap check let through. Such calls used to return wrong distances (for example float32 17.78 instead of 21.21 at `kernel_size=31`, `h=0.18`).

  `distance_transform` and `DistanceTransform` now reject an `h` above a bound that depends on `kernel_size` and the working precision: in 2-D `h < 0.5624` at `kernel_size=3` (`0.5626` in `float64`) and about `0.50` at larger kernels, in 3-D `h < 0.4070` at `kernel_size=3` (`0.4071` in `float64`) and about `0.38` at larger kernels. The bound leaves the off-centre kernel taps a margin below 1 for the rounding of the working precision. Such an `h` used to return 0, as if on the foreground, at background pixels whose window was full of reached pixels. ([#5961](https://github.com/kornia/kornia/pull/5961))
- `ImageStitcher` now raises `RuntimeError` naming the sample and its match count when a sample has fewer than 4 matches (sample 0 when nothing matched), and naming the index when a match's batch index is outside the batch. Before, a sample with 1 to 3 matches, or an unmatched sample ahead of a matched one, raised `ValueError` from RANSAC (`estimator="ransac"`) or `AssertionError` from `find_homography_dlt` (`"vanilla"`); unmatched samples at the end of a batch failed later with a `grid_sampler` `RuntimeError`; a batch without any match raised a `RuntimeError` that named no sample; and a batch index outside the batch raised one of these errors, depending on the indexes. `HomographyTracker.track_next_frame` called without a previous homography (before `forward()` or `match_initial()` found one) now raises kornia's `BaseError` naming the missing previous homography; before, it raised `UnboundLocalError`. It also raises that error when called right after a target is assigned directly, because the previous homography maps the old target's coordinates; before, it tracked the new target with that homography. ([#5965](https://github.com/kornia/kornia/pull/5965))
- `extract_tensor_patches`, `combine_tensor_patches`, `compute_padding`, `ExtractTensorPatches` and `CombineTensorPatches` reject a `window_size` or `stride` component below 1 up front with `ValueCheckError` (a `BaseError` and a `ValueError`) naming the argument, also under `python -O`. A zero or negative `stride` used to fail later with `ZeroDivisionError` or a `RuntimeError` from `unfold`/`fold`, and the two module classes accepted it at construction; `combine_tensor_patches` failed the same way for a non-positive `window_size`, or raised `AssertionError` when the stride exceeded it; `compute_padding` raised `ZeroDivisionError` for a zero stride and returned `(0, 0, 0, 0)` for a negative stride or a non-positive `window_size`. `extract_tensor_patches` still raises a `ValueError` for a non-positive `window_size`. ([#5969](https://github.com/kornia/kornia/pull/5969))
- `elastic_transform2d` and `RandomElasticTransform` no longer clamp the displaced sampling grid to `[-1, 1]` before `grid_sample`, so `padding_mode` now applies to coordinates displaced outside the image. Previously `"zeros"`, the default of both, and `"reflection"` repeated the edge pixel there (`"zeros"` at half intensity, a quarter in the corners, with the default `align_corners=False` and bilinear sampling); now those samples are zero-padded or reflected as requested. Default outputs therefore change near the image border wherever the displacement points outward, and so do the gradients through those samples. `"border"` output is unchanged. This correctness fix ships without a deprecation window. ([#5987](https://github.com/kornia/kornia/pull/5987))
- `YuNet(phase, pretrained=False)` returns the randomly initialised model in training mode; it returned it in evaluation mode. Call `.eval()` before inference, for example after loading your own weights. `FaceDetector`, which loads the weights, is unaffected. ([#6014](https://github.com/kornia/kornia/pull/6014))
- `HardNet`, `HardNet8`, `HyNet`, `SOSNet`, `TFeat`, `OriNet`, `LAFAffNetShapeEstimator`, `KeyNet`, `KeyNetDetector` and `DeFMO` default to `pretrained=True`. Constructing one downloads its weights (once) and returns the trained model in evaluation mode; for `KeyNetDetector` that is its `KeyNet` network, while the detector itself keeps its mode, so an `ori_module` or `aff_module` you pass is not switched. They defaulted to random weights, returned in evaluation mode with BatchNorm, where the network has any, on its initial statistics. Pass `pretrained=False` for random weights. An architecture argument the checkpoint does not fit now needs `pretrained=False` as well: with the default, `HyNet` with another `is_bias` or `dim_desc`, and `KeyNet` or `KeyNetDetector` with a `keynet_conf` of another `num_filters`, `num_levels` or `kernel_size`, raise `RuntimeError` from the strict weight load, and `HyNet(is_bias_FRN=False)` raises `ValueError` before downloading the weights, which would load but leave their trained FRN biases unused.

  Randomly initialised models are returned in training mode: `pretrained=False` on those classes, `SOLD2(pretrained=False)`, `SOLD2_detector(pretrained=False)` and `LoFTR(pretrained=None)`. They were returned in evaluation mode. Call `.eval()` before inference, and after loading your own weights into one of them. A random `LoFTR` in training mode runs its supervised training branch, so its forward raises without ground-truth matches until `.eval()` is called. ([#6017](https://github.com/kornia/kornia/pull/6017))
- `kornia.models.qwen25.Qwen2VLVisionTransformer` and the `kornia.models.qwen25.qwen2_vl` module with its blocks (`Qwen2VLPatchMerger`, `Qwen2VLRotaryEmbedding`, `Qwen2VLVisionAttention`, `Qwen2VLMLP`, `Qwen2VLVisionBlock`) are removed, so importing them now raises `ImportError`. They shipped in v0.8.3 and v0.9.0rc1 as a randomly initialised placeholder that was not the Qwen2.5-VL architecture (a plain ViT with a Conv2d patch embedding, LayerNorm, global attention and no positional information), and no weights existed for them. Use `kornia.models.qwen25.Qwen25VisionTransformer`, which takes a `Qwen25VisionConfig` and loads the released Qwen2.5-VL checkpoints. ([#6025](https://github.com/kornia/kornia/pull/6025))
- `kornia.models.sam3` is removed: `import kornia.models.sam3` raises `ModuleNotFoundError`. It shipped in v0.8.3 and v0.9.0rc1 as randomly initialised Phase 1–2 modules of #3406 (an image encoder that was a plain single-scale ViT, and a prompt encoder and mask decoder whose masks ignored point and box prompts), with no weights and no loader. SAM 3 is a different model under Meta's SAM License; a port would be a separate decision. ([#6029](https://github.com/kornia/kornia/pull/6029))
- `DepthEstimation` called on a `(B, 3, H, W)` tensor returns depth maps of shape `(B, 1, H, W)`, kornia's image layout; it returned `(B, H, W)`. A list of `(3, H, W)` images still returns `(1, H, W)` items. The graph exported by `to_onnx(include_pre_and_post_processor=True)` from a `DepthAnythingONNXBuilder` model takes an image that is already resized, such as `model.resize(image)`; it took an image of any size and squashed it to 518 x 518. The resize keeps the aspect ratio, so its output size depends on the input size, and it runs in torch before the graph. ([#6030](https://github.com/kornia/kornia/pull/6030))
- `VisualPrompter.predict(keypoints=...)` without `keypoints_labels` raises; the keypoints were ignored and the prediction was the one without prompts. `VisualPrompter.set_image` takes one image, `(3, H, W)` or `(1, 3, H, W)`, and raises for a batch: it accepted `(B, 3, H, W)` with B > 1, after which `predict()` without prompts returned one prediction per image and `predict` with keypoint or `Boxes` prompts raised. Use `Sam.forward` to segment a batch of images with per-image prompts. ([#6040](https://github.com/kornia/kornia/pull/6040))
- In the `EdgeDetector` that `EdgeDetectorBuilder.build("dexined")` returns, `model[0]` is now `nn.Sequential(RgbToBgr(), Normalize(...))`, and the normalisation is `model[0][1]`, with the BGR mean `[103.939, 116.779, 123.68] / 255`. `model[0]` was the `Normalize` module itself, with the ImageNet RGB mean `[0.485, 0.456, 0.406]`, so code that reads `model[0].mean` or `model[0].std` now raises `AttributeError`; read `model[0][1].mean` and `model[0][1].std` instead. DexiNed stays at `model[1]`, so a saved `EdgeDetector` state dict keeps its `model.1.*` keys. ([#6042](https://github.com/kornia/kornia/pull/6042))
- Some vision-language calls of `kornia.models` that ran now raise:

  - `SigLip2Config.from_name` and `SigLip2Builder.from_name` give the supported checkpoints (`google/siglip-base-patch16-224`, `google/siglip2-base-patch16-{224,256,384,512}` and `google/siglip2-large-patch16-{256,384,512}`) their 64 text positions, where they gave 512: a random-init model built from one of these names takes at most 64 tokens, and longer `input_ids` raise. Other names keep 512 positions.
  - SigLip2 honours `hidden_act`: `"gelu_pytorch_tanh"` and `"gelu_new"` give tanh GELU, `"gelu"` exact GELU, and any other value raises `ValueError` when the model is built. `hidden_act` was ignored (every MLP used exact GELU), so a config with `hidden_act="quick_gelu"` or `"relu"` built and ran.
  - With the legacy TorchScript ONNX exporter (`torch.onnx.export(..., dynamo=False)`), exporting a `NaFlex` from an example input whose patch grid is smaller than its position table along an axis raises `UnsupportedOperatorError`, because such a grid is now resized with antialiasing (`aten::_upsample_bilinear2d_aa`). It exported a plain bilinear resize, whose values were wrong at that size. An example grid at least as large as the table along both axes still exports, and its graph is exact for such grids only. The dynamo exporter (`dynamo=True`) exports the antialiased resize; with dynamic H and W, export from an example whose grid is smaller than the table along both axes, as the `NaFlex` note says.

  Some vision-language outputs of `kornia.models` change with no error, so weights fine-tuned with an earlier kornia give other outputs:

  - SigLip2's MLPs use the tanh approximation of GELU by default (`hidden_act="gelu_pytorch_tanh"`, the new config default), as the SigLIP 2 checkpoints do, where every MLP used exact GELU; `hidden_act="gelu"` keeps exact GELU.
  - `SigLip2ImagePreprocessor` and its `from_config` resize bilinearly by default, as the SigLIP 2 checkpoints do (`"resample": 2` in their `preprocessor_config.json`), where they resized bicubically. That includes the SigLIP 1 checkpoint `google/siglip-base-patch16-224`, whose preprocessor resizes bicubically: pass `interpolation="bicubic"` for it.
  - `PaliGemma` follows Gemma: it scales the text embeddings by `sqrt(hidden_size)`, its embedding's padding row is the new `PaliGemmaConfig.pad_token_id` (default 0), its MLP uses tanh GELU, and `GemmaRMSNorm` multiplies by `1 + weight` in float32 before casting; none of these held before. The padding row was `ignore_index = -100`, which wrapped to row `vocab_size - 100`, a real token whose embedding was zeroed and received no gradient.
  - `NaFlex` resizes its position table to the image's patch grid whenever that grid is not the table's own square grid, with bilinear interpolation as transformers' SigLIP 2 does, antialiased when the grid is smaller than the table along an axis (transformers always antialiases, which changes nothing when no axis shrinks). It added a table unresized in raster order whenever the patch count matched (4.27 off for an 8 x 32 grid and a 16 x 16 table), and did not antialias a downsampled axis (0.81 off).

  ([#6044](https://github.com/kornia/kornia/pull/6044))
- `equalize`, `equalize3d` and `equalize_clahe` now raise `TypeCheckError` (a `TypeError`) for an integer or boolean image, with a message that names the dtype and the conversion `image.float() / 255`, as the `[0, 1]` adjustment functions do since #5983; the check is skipped under `disable_checks()`. Before, their `[0, 1]` range check let through an integer or boolean mask whose values are all 0 or 1: `equalize` and `equalize3d` returned the same result as for the floating-point mask and now raise, so convert such a mask with `.float()` first; `equalize_clahe` returned values that did not match the floating-point mask's (by up to almost 1.0 at `clip_limit=0`), and a boolean mask made it raise a raw `NotImplementedError`. A `uint8` image with values above 1 used to raise a `RuntimeError` and now raises the `TypeCheckError`, which `except RuntimeError` does not catch. Floating-point input is unchanged, and out-of-range floating-point input still raises the `RuntimeError` naming the range. ([#6047](https://github.com/kornia/kornia/pull/6047))
- Vision-language calls of `kornia.models` that ran now raise, or read a name or a mask differently:

  - `SigLip2Config.from_name`, `SigLip2Builder.from_name` and `SigLip2Builder.from_pretrained_hf` raise `ValueError` for a name outside their supported list whose architecture its parts cannot give: a `google/` name, such as `google/siglip2-so400m-patch14-384` or the SigLIP 1 `google/siglip-large-patch16-256`; a SigLIP 1 repository under another owner, such as `someone/siglip-large-patch16-256`, unless it is a fine-tune or mirror of `google/siglip-base-patch16-224`; and a repository part with the word `so400m`, `giant` or `naflex`, or `patch<n>` for a patch size other than 16. `from_name` returned a base or large patch-16 SigLIP 2 model of the wrong shape or vocabulary for these names, and `from_pretrained_hf` failed on loading its weights.
  - These calls read the architecture of a name outside their supported list from the whole words of its repository part only, a word being a run of letters and digits: `large-models/siglip2-base-patch16-224` and `someone/siglip2-base-patch16-224-enlarged` are base, where any `large` in the name made them large.
  - `MoonViTAttention`, `MoonViT` and `KimiVLModel` read a 2-D attention mask as `(B, N)` padding and a 3-D one as `(B, N, N)`, one mask per sample, as `SigLip2Attention` does, whatever the batch size, and raise for one whose leading dimension is neither the batch size nor 1. A per-head `(H, N, N)` mask and an `(N, N)` mask shared by the batch were passed to `scaled_dot_product_attention` as they were, so they raise now unless `B` equals `H` or `N`, where they are read per sample: pass them as `(1, H, N, N)` and `(1, 1, N, N)`.

  ([#6051](https://github.com/kornia/kornia/pull/6051))
- Some calls of the vision backbones of `kornia.models` that ran now raise:

  - `VisionTransformer` and `SigLip2VisionEmbeddings`, and so `SigLip2VisionModel`, `SigLip2Model` and `PaliGemma`, raise `ValueError` for an input whose height and width are not both `image_size`, and `VisionTransformer` also for a channel count other than `in_channels`. With a `backbone`, `VisionTransformer` checks the backbone's token grid instead: it raises `ValueError` when the grid is not the one the backbone gives for an `image_size` input, which the position embeddings are built for, and inputs of other sizes that give that grid, through adaptive pooling for example, keep working. An input whose patch count happened to fit the position table returned a result with misplaced position embeddings or dropped columns: for `VisionTransformer(image_size=224, patch_size=16)`, `(1, 3, 112, 448)` added the 14 x 14 positions to a 7 x 28 patch grid and `(1, 3, 224, 239)` dropped its last 15 columns. Other sizes and a channel mismatch raised `RuntimeError` inside the model, for example `(1, 3, 240, 240)` and `(1, 1, 224, 224)`; the `ValueError` of `VisionTransformer` fired only when both the size and the channel count were wrong.
  - `TinyViT` raises `ValueError` for an input whose token grid is not the one its attention stages are built for, that of an `img_size` x `img_size` image; sides from `img_size - 7` to `img_size` round to that grid and keep working. An input with the same number of tokens on another grid returned a result with its tokens read as the wrong grid: at `img_size=224`, `(1, 3, 112, 448)` returned `(1, 1000)` from a 14 x 56 grid read as 28 x 28, and with `mobile_sam=True` at `img_size=1024`, a `(1, 3, 512, 2048)` input returned feature maps. Other sizes raised `RuntimeError`, for example `(1, 3, 224, 256)`.
  - `TinyViT.from_config("21m", ...)` builds the `"in1k_384"` and `"in1k_512"` checkpoints with 12- and 16-px windows (see the bug fixes), so a state dict saved from such a model built before, with 7-px windows, no longer loads into the model the same call builds: `load_state_dict` raises `RuntimeError` for the size of every attention bias, `(6, 49)` against `(6, 144)` in `layers.1.blocks.0.attn.attention_biases` at 384. Pass `window_sizes=[7, 7, 14, 7]` to build the model that state dict belongs to.

  ([#6056](https://github.com/kornia/kornia/pull/6056))
- `HomographyTracker` returns and stores every homography, `no_match()`'s included, in float32 for a `float16` or `bfloat16` target and in the target's dtype otherwise, whatever the dtype of the frame, of the matcher's keypoints or of a `previous_homography` assigned from outside. A homography found by matching used to have the keypoints' dtype: a `float64` tracker whose matcher returns float32 keypoints returned float32 homographies, and a float32 tracker given float64 frames, by a matcher that returns keypoints in the frame's dtype, returned float64 ones. `no_match()` returned the half dtype. The tracker now also tracks with `float16` and `bfloat16` matchers. The second frame raised: in `warp_perspective`, which got the half frame with the float32 inverse of the first frame's float32 homography, or in `linalg.inv` for a half `previous_homography`. The prior is now inverted and the frame warped in float32, and the fast matcher gets the warped frame in the frame's dtype. Half keypoints are widened to float32 before RANSAC; a bfloat16 fit was rounded to bfloat16 and moved the tracked points by pixels. The caller's autocast no longer runs the tracker's own warp, keypoint mapping and `ransac` fit in half precision; under `bfloat16` autocast the keypoint mapping was up to 7.6 px off on a 600 px frame, and a `RANSAC(local_optimization="dlt")` fit was several pixels off. A float32 tracker whose matcher returns float64 keypoints, or with a float64 `previous_homography` assigned from outside, raised in `track_next_frame`. ([#6058](https://github.com/kornia/kornia/pull/6058))
- `ONNXSequential` given `target_opset_version` or `target_ir_version` without `auto_ir_version_conversion=True` now
  converts the models; the targets used to be ignored, and the call ran on the models as loaded. Where the conversion
  fails, the call now raises its error: the ONNX version converter's where it has no adapter for an operator (for example
  `Resize` or `Pad` from opset 18 to 17), `onnx.checker`'s where the converted graph is invalid (for example
  `Unrecognized attribute: noop_with_empty_axes for operator ReduceMean` from opset 18 to 17), or ONNX Runtime's where it
  does not support the target (an IR version above its own). ([#6134](https://github.com/kornia/kornia/pull/6134))
- Two `to_onnx` exports change in ways a caller can see:

  - An `EdgeDetector` whose `ResizePostProcessor` resizes with `interpolation_mode="area"` now raises a `ValueError` in
    `to_onnx` with the pre- and post-processor, and so does a direct `torch.onnx.export(detector, ...)` on the
    TorchScript exporter and, from torch 2.9, on the dynamo exporter (inside that exporter's own error). It used to export
    a graph that returned the maps at the pre-processor's size. `"area"` resizing is adaptive average pooling, which
    cannot be exported to the input's size: build the post-processor with `"bilinear"`, `"bicubic"`, `"nearest"` or
    `"nearest-exact"` to export the detector.
    Eager `forward` with `"area"` is unchanged.
  - `ObjectDetector.to_onnx(include_pre_and_post_processor=False)` of an RT-DETR model names its outputs `pred_logits`
    and `pred_boxes`. The logits used to be the graph's `output`, and the boxes came out under an exporter-generated
    name, such as `2814` from the TorchScript exporter, so code that reads `output` by name, for example
    `session.run(["output"], ...)`, is now refused by ONNX Runtime. Read `pred_logits` instead, or pass `output_names`
    to `to_onnx` to choose both names.

  ([#6135](https://github.com/kornia/kornia/pull/6135))
- `BoxFiltering.to_onnx()` without a `confidence_threshold` on the legacy (TorchScript) exporter used to export a second float64 input, `onnx::Cast_1`, for the threshold, which the caller fed. The graph now has the boxes as its only input and filters at the eager default of `0.0`, so feeding two inputs raises; build the filter with `confidence_threshold` to export a different threshold. ([#6159](https://github.com/kornia/kornia/pull/6159))
- `zca_mean` returns, and `ZCAWhitening.fit` stores, the mean and transforms in float32 for float16 and bfloat16 inputs; previously they were in the input dtype, whose rounding could corrupt rank-deficient whitening. `linear_transform`, `zca_whiten`, and `ZCAWhitening` still return the input dtype, computing half-input transforms in float32. Callers applying `zca_mean` results directly must promote half inputs to float32 or use `linear_transform`. Explicitly casting fitted buffers to half precision still discards their accuracy. ([#6212](https://github.com/kornia/kornia/pull/6212))
- `blur_pool2d`, `max_blur_pool2d`, `edge_aware_blur_pool2d`, `BlurPool2D`, `MaxBlurPool2D` and `EdgeAwareBlurPool2D`
  raise a `TypeCheckError` (a `TypeError`) for an integer or bool input, naming the dtype. The blur pools are network
  layers and expect a floating feature map; unlike the image filters (#5669) they do not promote integer input. Before,
  the binomial kernel was cast to the input's dtype, which truncated every tap to 0. On CPU, `BlurPool2D` and
  `MaxBlurPool2D` returned an all-zero integer tensor without an error, and the functions raised a `RuntimeError` dtype
  mismatch. On CUDA, every entry point raised a `RuntimeError` from cuDNN or the max pool. `except RuntimeError` no
  longer catches the error. The check is skipped under `disable_checks()` in Python; scripted calls always validate,
  as in Kornia's core checks. `BlurPool2D` and `MaxBlurPool2D` remain scriptable. Floating input is unchanged. ([#6270](https://github.com/kornia/kornia/pull/6270))
- Kornia's minimum supported PyTorch version rises to 2.5.1. Previously the declared floor was
  2.0.0; PR-time CI has only ever exercised 2.5.1 and newer, and this same change retires the
  scheduled-CI legs that tested anything older.
  Installing kornia now requires `torch>=2.5.1`; a `torch` install below that version no longer
  satisfies the dependency, and `pip`/`uv` will refuse to resolve it. (#4197)
- Non-maxima suppression applies one border rule at every window size. `NonMaximaSuppression2d` /
  `nms2d` with a window larger than `(7, 7)`, and `NonMaximaSuppression3d` / `nms3d`, no longer report
  maxima inside the `(k - 1) // 2` border strip. Previously the general path replicate-padded its input,
  so a position whose window ran off an edge was judged against duplicated copies of the edge values,
  which both fabricated plateaus that suppressed genuine maxima and hid genuine neighbours that should
  have suppressed spurious ones; the `(3, 3)`, `(5, 5)` and `(7, 7)` paths meanwhile rejected the strip
  outright, so the two disagreed about the same pixel. The explicit paths' rule is now the rule
  everywhere. The kornia detectors that call `nms2d` already reject that strip: ALIKED zeroes its
  `nms_radius` border by hand after the call, DISK and XFeat call it with a `(5, 5)` window by default,
  whose explicit path already had this rule, and `MultiResolutionDetector` zeroes a 15 px border in its
  output, which is wider than the strip of its default `(15, 15)` window. No detector output changes with
  its default configuration. (#4239, #4242)
- Removed `kornia.to_tensorflow()`, `kornia.to_jax()` and `kornia.to_numpy()`, along with the
  multi-framework-support advertising in the README and the docs landing page. These functions used
  to lazily transpile the library to TensorFlow, JAX or NumPy through the third-party `ivy` package
  (an optional `dev`/`docs` dependency, now also removed). Testing in September 2026 found the
  integration unreliable across all three targets — a checkout-path crash in Ivy's module walker, a
  torch/triton segfault when `transformers` is also installed, an undocumented `flax` requirement for
  JAX, and a `jaxlib`/`numpy` API-compatibility break — against Ivy's latest release (1.0.0.5, June
  2025), with the upstream project showing little ongoing maintenance. The three names still
  import, at the top level and from `kornia.transpiler`, so `from kornia import to_jax` and
  `hasattr(kornia, "to_jax")` keep working; calling any of them raises a `RuntimeError` that says
  what was removed and links the page below. See `get-started/multi-framework-support` for what
  was tested and why. (#4196)
- Removed the unused `preprocess_boxes` helper, which had no public export or caller. (#4181, #4321)
- The shape guards of `normalize_homography` and `denormalize_homography` now reject everything but a
  `(3, 3)`/`(B, 3, 3)` matrix, and `normalize_homography3d`'s everything but a `(4, 4)`/`(B, 4, 4)` one (#3999).
  A rank-4 input used to pass the guard and come back with its rank unchanged —
  `normalize_homography(eye(3).expand(2, 4, 3, 3), (4, 5), (8, 9))` returned a `(2, 4, 3, 3)` matrix and now raises
  `ValueError`. A wrong-sized input, such as a `(B, 4, 4)` to the 2-D functions, used to pass the guard and fail
  later inside `matmul`; it now raises at the guard, and `normalize_homography3d`'s message names `Bx4x4` instead of
  `Bx3x3`. Unbatched `(3, 3)`/`(4, 4)` matrices are still accepted and still promoted to a leading batch of 1.

  One side effect reaches tracing callers: the rewritten guard evaluates its shape comparison unconditionally, where
  the old `or`-form short-circuited past it for a rank-3 input, so `torch.jit.trace` of any of the three functions
  now emits a `TracerWarning` about converting a tensor to a Python boolean that it did not emit before. The guard
  is a static check and the traced graph is unchanged — the warning is noise, not a correctness signal.
- `AUTUMN` is dropped from `kornia.color.__all__` (#4143). The class was deprecated in 0.7.2 in favour of
  `ColorMap(base='autumn')` and removed by #3432, which left the `__all__` entry behind. Since that removal
  `from kornia.color import AUTUMN` has raised `AttributeError` and `from kornia.color import *` has failed on
  it, so nothing that works today stops working — the entry named a binding that no longer existed. Callers
  still reaching for the class should use `ColorMap(base='autumn')`.
- `DescriptorMatcherWithSteerer(normalize=True)`, `DiscreteSteerer.steer_descriptions(normalize=True)` and the
  `MKD` descriptors, like `SIFTDescriptor` and `HardNet`, L2-normalise a float16 input in float32 and cast back,
  so an all-zero float16 descriptor normalises to zero instead of NaN (the 1e-12 guard underflows in float16) and a
  descriptor whose norm sits below the smallest float16 normal still normalises to one rather than being clamped.
  `PatchDominantGradientOrientation`, and with it `LAFOrienter`, take the gradient magnitude and orientation of a
  float16 patch through the same float32 step `SIFTDescriptor` uses, so a flat patch yields a finite angle instead
  of a NaN LAF -- which reached a *filled* detection slot in a float16 `SIFTFeature` pipeline and poisoned that
  descriptor row. `SIFTDescriptor(rootsift=True)` and `DenseSIFTDescriptor(rootsift=True)`
  compute the RootSIFT square root in float32 for a float16 input: the float16-representable guard would have
  read every empty bin as `sqrt(6.1e-5)` and biased the descriptor norm to ~1.004. Float32 and float64 outputs are
  unchanged.
- `KimiVLBuilder.from_pretrained_hf()` and `SigLip2Builder.from_pretrained_hf()` cache their
  checkpoint where every other kornia checkpoint lives. `cache_dir=None` used to mean the HuggingFace
  cache (`~/.cache/huggingface/hub/models--<owner>--<name>/snapshots/<sha>/model.safetensors`) and now
  means torch's hub directory (`<torch hub dir>/checkpoints/<owner>--<name>--model.safetensors`); an
  explicit `cache_dir` used to hold the same `models--<owner>--<name>/…` tree and now holds the flat
  `<owner>--<name>--model.safetensors` file. Either way the first `from_pretrained_hf()` call after
  upgrading re-downloads the checkpoint (854 MB for KimiVL, ~1.5 GB for SigLIP2) even for users who
  already had it. (#4293)
- `LocalFeatureMatcher` no longer matches the zero-LAF slots with which a fixed-shape detector pads an under-filled
  result, and now forwards its documented `mask0`/`mask1` inputs to feature extraction. `nn` and `mnn` callers can
  therefore receive fewer correspondences: identical descriptors sampled at padded origin frames are no longer
  reported as matches. Three-dimensional `(B, H, W)` masks keep working and are promoted to the detectors'
  `(B, 1, H, W)` form; four-dimensional masks are forwarded unchanged. Because the masks now reach the detector,
  they are subject to its check: a mask that is not at the image's spatial size, which used to be ignored, is
  rejected. A floating-point mask changes meaning in `ScaleSpaceDetector`-based pipelines too; see the mask semantics entry under **Bug fixes**.
- `MKDDescriptor` runs in `float16` and `bfloat16`. Its gradient-embedding and spatial-encoding stages were held in
  a plain `dict`, so `.to(device, dtype)` never reached their buffers and a half-precision input failed at the
  whitening matmul with `expected m1 and m2 to have the same dtype`. They are now an `nn.ModuleDict`; float32 and
  float64 outputs are byte-identical, and `state_dict()` gains the stages' buffer keys (`feats.<parametrization>.*`).
  A state dict saved by an earlier release still loads with `strict=True`: those buffers are derived from the
  constructor arguments, and when a state dict has no `feats.*` key at all they are filled in from the module being
  loaded into (one that has some and lost one is reported by `strict=True`, as for any other missing key). The
  reverse is not covered: a state dict saved by this release carries the `feats.*` keys and fails `strict=True` on
  an earlier kornia, which does not know them; load it with `strict=False` there.
- `MultiResolutionDetector` and `KeyNetDetector` change their output: padded slots now read as a zero response and
  a zero LAF instead of `torch.finfo(dtype).min / 2` and an arbitrary border coordinate; the previously inert `mask`
  argument now suppresses detections, and must be `(1 or B, 1, H, W)` at the image size -- a `(B, H, W)` mask
  passed directly to either detector, which was accepted only because it went unread, is rejected; half-precision
  input yields half-precision LAFs; the returned shape is always `num_features`; a multi-channel response is rejected instead of
  producing invalid or duplicate LAFs; a negative `score_threshold` is rejected with `ValueError`; and `detect`
  enforces its documented `(1, C, H, W)` input. The `mask` of both detectors is now "where a detection may be": a
  boolean or integer mask is binary (a 0/255 mask no longer scales the responses by 255), a floating-point mask
  weights the scores, it is resampled conservatively so a thin zero region survives the coarse levels, and it is
  applied to the non-maxima-suppression output, so its edge cannot manufacture maxima (#4102). A mask whose dtype
  differs from the image no longer promotes the response dtype, and a mask that is not `(1 or B, 1, H, W)` for the
  image is rejected instead of stretched or broadcast onto the wrong axis -- that includes an image-shaped
  `(1, C, H, W)` mask and a coarse `(N, H/8, W/8)` one, which `LocalFeatureMatcher` used to accept and silently
  ignore (the old docstring said "same shape as the input image") and now forwards to the detector's check.
  `ScaleSpaceDetector` additionally stops returning its own top-K sentinel as a detection for a batch, no longer
  returns a frame for a candidate its border check rejected, and re-checks a sub-pixel-refined centre against the
  mask; a short result is sorted in both detectors. `detect_features_on_single_level`'s new `mask` parameter is
  keyword-only, and it now requires the response map to have the level's spatial size: a valid-convolution
  response net that returned a smaller map had every keypoint decoded one pixel off its peak, since a response
  index is read as a level pixel with no offset, and is rejected with a message rather than silently misplaced.
  See the entry under **Bug fixes** (#4089, #4090, #4091) for the details and the
  migration.
- `PatchSequential` now preserves the batch size in both padding modes. `same` preserves the input
  spatial size and `valid` keeps only the complete, centred grid region. Previously, a `(2,3,8,8)`
  input on a `(3,3)` grid produced `(2,3,7,7)` with `same` (now `(2,3,8,8)`) and `(2,3,8,8)` with
  `valid` (now `(2,3,6,6)`). With `valid`, a `(2,3,6,8)` input on a `(4,4)` grid changed the batch
  size to produce `(3,3,6,6)`; it now produces `(2,3,4,8)`. Geometric children may move temporarily
  padded zeroes into the retained image. `restore_from_patches` now defaults to `self.grid_size`
  instead of `(4,4)` and raises `ValueError` for a mismatched patch count instead of inferring a
  different batch size or failing later in reshape. `forward_parameters` accepts both image
  `(B,C,H,W)` and patch `(B,N,C,h,w)` shapes; parameter generation now covers `B*N` patch rows rather
  than `B*C`, so fixed-seed outputs and RNG consumption can change. Refs #4421. (#4460)
- `SegmentationModelsBuilder.build()` no longer imports `segmentation_models_pytorch` (smp). It used
  to take `model_name`, `encoder_name`, `encoder_weights`, `in_channels`, `classes`, `activation` and
  `**kwargs`, import smp lazily, instantiate the smp architecture and look up the encoder's
  preprocessing parameters itself. It now takes a constructed `nn.Module` and the encoder's
  preprocessing-parameter dictionary (`build(model, preproc_params=None, name="segmentation_model")`),
  where `preproc_params` is what `smp.encoders.get_preprocessing_params(encoder_name)` returns; build
  the smp network and fetch its parameters yourself. Kornia no longer imports smp anywhere, and
  `kornia.core.external.segmentation_models_pytorch` is gone. `SemanticSegmentation` gained the
  `__init__(model, pre_processor, post_processor, name=None)` its siblings have, so the container the
  builder returns can be instantiated (it was abstract before). The `input_range: [0, 255]`
  preprocessing step is now `kornia.enhance.Rescale(255.0)`, a multiply by 255, instead of a
  `Normalize` by a stored `1/255`: bfloat16 rounds that reciprocal to a ~254.0 multiplier (0.5 mapped
  to 127.0 instead of 127.5), and float32 outputs of that step move by at most one ulp. (#4301)
- `create_meshgrid` and `create_meshgrid3d` return different normalized coordinates at `float16`
  and `bfloat16`. They used to evaluate `(xs / (size - 1) - 0.5) * 2` as three half-precision ops
  in eager, rounding after each, and to materialize the half ramp before the graph-capture branch
  widened it; `torch.compile` instead computed the chain in `float32` and rounded once, and folded
  the narrow-then-widen round trip away so that above `2 ** p` the two paths divided different
  numerators. A half-precision grid therefore changed the moment a model was wrapped in
  `torch.compile` -- by up to 0.0117 in the `[-1, 1]` range at `bfloat16` width 300 and 0.0156 at
  width 6000. A normalized half-precision grid is now built and normalized in `float32` and
  narrowed once, so eager, traced and compiled grids agree; the eager values move onto the
  correctly rounded side, which is the numeric change. `float32`, `float64` and integer grids and
  the pixel ramp are bit-for-bit unchanged, since none of them widen. On CUDA at `float16` up to
  one ulp of the coordinate dtype can still differ, because ATen's and triton's `float32` division
  are not bit-identical; `bfloat16` absorbs it and the CPU kernels agree exactly. The Linux CPU
  half-precision baselines move with it. `HomographyWarper` builds its precomputed grid in `float32`
  and its on-the-fly grid in the input dtype, and the two now agree, which fixes
  `test_homography_warper` at both dtypes and `test_translation[shape1]` at `float16`. At `bfloat16`
  `test_translation[shape0]` and `[shape2]` join the `shape1` and `shape3` entries already recorded:
  that test's `atol=1e-4` is out of reach for all four of its shapes there, and the two that passed
  did so only because the old grid's larger error round-tripped through `grid_sample`'s `bfloat16`
  unnormalize as an exact pixel index. Refs #4030. (#4231)
- `ellipse_to_laf` no longer raises on a degenerate ellipse. One whose `a` or `c` is `0` after rounding to the input
  dtype makes the matrix being inverted singular; the batched `torch.inverse` it now replaces (see **Bug fixes**) failed the
  whole call with `linalg.LinAlgError`, and the closed form returns non-finite values in that LAF instead, leaving
  the other rows of the batch usable. Callers that relied on the exception to reject bad input -- reading
  Oxford-format `.ellipse` files, say -- should test the result with `kornia.feature.laf_is_valid`, since an invalid
  LAF otherwise propagates silently through `get_laf_scale` and into matching. Detecting the degenerate rows up front
  would cost a device synchronisation and the `fullgraph=True` capture, so it is left to the caller. The in-repo
  affine estimators handle invalid candidates before nonlinear LAF operations. `PatchAffineShapeEstimator` computes
  half-precision gradients, Gaussian weights, moments, and normalization in `float32`, preserving valid anisotropic
  shapes that would otherwise underflow; `LAFAffineShapeEstimator` sanitizes non-positive-definite ellipse
  coefficients before conversion; and both it and `LAFAffNetShapeEstimator` use a finite fallback with finite
  backward. The fallback is the input LAF when `preserve_orientation=True` and its upright form otherwise; a finite
  zero-scale input keeps its center and zero affine block.
- `extract_patches_from_pyramid` now samples ordinary-sized inputs once from a packed pyramid atlas instead of
  sampling every LAF at every pyramid level. A statically selected levelwise fallback keeps atlases larger than
  128 MiB from becoming a memory hazard. LAFs whose nominal level is coarser than the last usable pyramid level now
  sample that actual coarsest level; they previously returned an all-zero patch. For `float16` and `bfloat16` inputs
  the whole pyramid -- padding, `pyrdown`, coordinate arithmetic and sampling -- runs in `float32` and the patches
  are cast back, on every device: this avoids the normalized-coordinate precision loss caused by the wider atlas,
  keeps the extraction off `replication_pad2d`/`grid_sampler_2d` kernels that old torch builds lack for half dtypes
  on the CPU (torch <= 2.5.1 would otherwise crash), and avoids recasting the full atlas on every sampling chunk.
  `extract_patches_simple` computes its sampling grid in `float32` for reduced-precision inputs on every device for
  the same reason -- its output matches the pyramid extractor's level-0 output again. Half-precision patches
  therefore change on CUDA and MPS as well as the CPU, and floating-point results are not byte-identical to the
  former levelwise implementation.
  The universal `float32` sampling is a real cost where the native half kernels were fine: on CUDA, `float16`
  `extract_patches_simple` is roughly 2x slower at high N, and peak CUDA memory rises at moderate N (the atlas plus
  the one-time upcast copy) even though the high-N peaks drop.
  Both extractors now reject an image/LAF batch-size mismatch at the boundary with a clear error: former releases
  raised an internal `grid_sample` error for most batch mixes, except a smaller LAF batch, where both extractors
  silently returned patches for only the first `min(B_img, B_laf)` images. A LAF on another device is moved once to
  the image, while a different LAF dtype is promoted with the image dtype for grid arithmetic rather than rejected or
  rounded down; this preserves mixed-precision autocast pipelines and subpixel LAF coordinates. Patches still return
  with the image's dtype and device. Both extractors chunk their sampling along N, in the spirit of `batched_forward`, so
  the folded `(B, N*PS, PS, 2)` grid and channel-scaled sampled chunk each stay within a fixed 64 MiB budget (remap
  intermediates can still make the total peak a multiple of that). Batched high-N extraction now peaks below even the
  pre-batching per-image loop (468 MiB
  against 495 MiB on `main` for B=8,
  N=4000, PS=32 on a 512x512 float32 image, where the unchunked atlas peaked at 1.24 GiB), eager throughput stays
  within ~5% of the unchunked atlas, and float32/float64 patches are bitwise identical to it; a `torch.compile`d
  call whose grid exceeds the budget splits into several `grid_sample` calls (~1.5x the unbounded atlas latency at
  B=8, N=2000 on CPU inductor, still ~10x faster than the per-level loop it replaces). Both extractors return an
  empty `(B, N, CH, PS, PS)` tensor for `B == 0` or `N == 0` instead of raising from inside `affine_grid` --
  `extract_patches_from_pyramid` with `B == 0` returned the empty result before the batched rewrite described under **Bug fixes**
  regressed it; every other empty combination raised on all prior releases. `extract_patches_from_pyramid` now
  also accepts a LAF on a different device than the image, as `extract_patches_simple` always has, and returns
  patches on the image's device; both extractors move the LAF to the image's device up front, so a cross-device
  call now samples on the image's device and matches the same-device result (`extract_patches_simple` previously
  built the full sampling grid on the LAF's device).
- `kornia.contrib.BoxMotTracker` is removed, together with the `kornia.contrib.boxmot_tracker` module.
  It wrapped the [boxmot](https://github.com/mikel-brostrom/boxmot) tracker zoo around a kornia detector,
  but called the boxmot 10.x API (`boxmot.DeepOCSORT(model_weights=..., device=..., fp16=...)`), and no
  boxmot release that installs alongside the torch versions kornia supports (`torch>=2.5.1`) provides
  it, so the class could not be instantiated on any supported stack (see
  [#4320](https://github.com/kornia/kornia/issues/4320)). Track with boxmot directly, feeding it the
  output of a kornia detector such as `RTDETRDetectorBuilder`. `kornia.core.external.boxmot` is gone
  with it. (#4301)
- `kornia.core.external.transformers` was removed; it was a `LazyLoader` handle no kornia code used.
  Previously `from kornia.core.external import transformers` gave a lazy proxy that imported
  `transformers` on first attribute access; that name no longer exists, so import `transformers`
  directly instead. (#4301)
- `kornia_rs>=0.2.0` is required; the floor used to be 0.1.9. kornia_rs 0.1.11 relocated its image I/O
  and 0.1.12/0.1.13 lack the libjpeg-turbo reader, so no single call site works across 0.1.9-0.1.14;
  rather than dispatch per version, kornia calls the current layout. Run `pip install -U kornia_rs` in an
  environment that pins an older one. (#4326)
- `nms2d(x, (1, 1))` and `nms3d(x, (1, 1, 1))` return `x` unchanged. A unit window has no neighbours,
  so nothing can be suppressed; the old 2-D result was an artifact of the zeroed centre tap of the
  convolution kernel summing to `0.0`, while the old 3-D path raised because of #4241. (#4242)


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
