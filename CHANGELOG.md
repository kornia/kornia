# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

****

## Unreleased

### Breaking changes

* The shape guards of `normalize_homography` and `denormalize_homography` now reject everything but a
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

* `MultiResolutionDetector` and `KeyNetDetector` change their output: padded slots now read as a zero response and
  a zero LAF instead of `torch.finfo(dtype).min / 2` and an arbitrary border coordinate; the previously inert `mask`
  argument now suppresses detections, and must be `(1, 1, H, W)` at the image size; half-precision input yields
  half-precision LAFs; the returned shape is always `num_features`; a multi-channel response is rejected instead of
  producing invalid or duplicate LAFs; a negative `score_threshold` is rejected with `ValueError`; and `detect`
  enforces its documented `(1, C, H, W)` input. The `mask` of both detectors is now "where a detection may be": a
  boolean or integer mask is binary (a 0/255 mask no longer scales the responses by 255), a floating-point mask
  weights the scores, it is resampled conservatively so a thin zero region survives the coarse levels, and it is
  applied to the non-maxima-suppression output, so its edge cannot manufacture maxima (#4102). A mask whose dtype
  differs from the image no longer promotes the response dtype, and a mask that is not `(1 or B, 1, H, W)` for the
  image is rejected instead of stretched or broadcast onto the wrong axis. `ScaleSpaceDetector` additionally stops
  returning its own top-K sentinel as a detection for a batch and no longer returns a frame for a candidate its
  border check rejected; a short result is sorted in both detectors. `detect_features_on_single_level`'s new `mask`
  parameter is keyword-only. See the entry under **Bug fixes** (#4089, #4090, #4091) for the details and the
  migration.

* `MKDDescriptor` runs in `float16` and `bfloat16`. Its gradient-embedding and spatial-encoding stages were held in
  a plain `dict`, so `.to(device, dtype)` never reached their buffers and a half-precision input failed at the
  whitening matmul with `expected m1 and m2 to have the same dtype`. They are now an `nn.ModuleDict`; float32 and
  float64 outputs are byte-identical, and `state_dict()` gains the stages' buffer keys (`feats.<parametrization>.*`),
  so a state dict saved by an earlier release loads with `strict=False`.

* `DescriptorMatcherWithSteerer(normalize=True)`, `DiscreteSteerer.steer_descriptions(normalize=True)` and the
  `MKD` descriptors normalise with an `eps` representable in the
  input dtype, so an all-zero float16 descriptor normalises to zero instead of NaN, the same guard `SIFTDescriptor`
  and `HardNet` gain in this release. `SIFTDescriptor(rootsift=True)` and `DenseSIFTDescriptor(rootsift=True)`
  compute the RootSIFT square root in float32 for a float16 input: the float16-representable guard would have
  read every empty bin as `sqrt(6.1e-5)` and biased the descriptor norm to ~1.004. Float32 and float64 outputs are
  unchanged.

* `LocalFeatureMatcher` no longer matches the zero-LAF slots with which a fixed-shape detector pads an under-filled
  result, and now forwards its documented `mask0`/`mask1` inputs to feature extraction. `nn` and `mnn` callers can
  therefore receive fewer correspondences: identical descriptors sampled at padded origin frames are no longer
  reported as matches. Three-dimensional `(B, H, W)` masks keep working and are promoted to the detectors'
  `(B, 1, H, W)` form; four-dimensional masks are forwarded unchanged. Because the masks now reach the detector,
  they are subject to its check: a mask that is not at the image's spatial size, which used to be ignored, is
  rejected. A floating-point mask changes meaning in `ScaleSpaceDetector`-based pipelines too, see below.

### Bug fixes

* Raise `NotImplementedError` instead of `RuntimeError` for an integral `src` on the empty-`dsize` paths of
  `warp_affine`, `warp_perspective`, `remap`, `warp_affine3d` and `warp_perspective3d`, matching what the non-empty path already raises from
  `grid_sample` (#4031). The degenerate path had its own explicit guard that fired first with a different exception
  type, so the exception a caller saw for the same invalid input depended on whether `dsize` had a zero dimension.
  The message is unchanged. `NotImplementedError` derives from `RuntimeError`, so `except RuntimeError` around these
  functions still catches it; only code matching on the exact type sees a difference.

* Make `yuv_to_rgb` the exact inverse of `rgb_to_yuv`, so an RGB → YUV → RGB round trip is now limited only by the
  input dtype instead of losing up to `1.36e-3` (in B, at `rgb = (1, 1, 0)`) at every precision, `float64` included
  (#4044). The inverse kernel was a separately rounded copy of the published BT.470-5 M/PAL inverse relations rather
  than the inverse of the rounded forward kernel kornia actually ships, and one of its literals — `2.029`, where the
  inverse of kornia's forward kernel is `2.03199968` — carried most of the error. `yuv_to_rgb` output therefore moves
  by up to `1.23e-4` in R, `9.13e-4` in G and `1.60e-3` in B over the documented YUV domain, and `yuv420_to_rgb` and
  `yuv422_to_rgb` move with it. The forward direction (`rgb_to_yuv`, `rgb_to_yuv420`, `rgb_to_yuv422`) is unchanged.
  Agreement of the inverse with the standard's own relations improves overall, from `1.54e-3` to `5.24e-4`, though R
  alone moves the other way (`1.54e-4` to `2.77e-4`). The `.. warning::` blocks that documented the defect on the six
  affected functions and classes are gone.

* Fix the inverted `SigLip2` attention-mask polarity, so every call that passes an `attention_mask` changes (#4043).
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

* Fix `find_essential` and `run_5point` raising `TypeError: all() received an invalid combination of arguments` on
  PyTorch before 2.2 (#4043). `run_5point`, `_solve_2x2_tikhonov_safe` and the scripted Nister helper each reduced
  over two axes at once with `Tensor.any(dim=(-2, -1))` / `Tensor.all(dim=(-1, -2))`, and multi-dimension `any`/`all`
  only landed in PyTorch 2.2, so the whole five-point solver was unusable on the declared `torch>=2.0` floor. The
  reductions now flatten the two trailing axes first, which is equivalent on every version.

* Fix `torch.jit.script` of `rgb_to_yuv`, `yuv_to_rgb`, `rgb_to_xyz` and `xyz_to_rgb` on older PyTorch (#4043). Their
  shared `kornia.color.utils._apply_linear_transformation` helper annotated its optional argument with the PEP 604
  `torch.Tensor | None` form, which the TorchScript compiler on the declared floor does not accept (reproduced on
  PyTorch 2.1.2, while PyTorch 2.9.1 compiles it), so scripting any of the four conversions failed. The annotation is
  `Optional[torch.Tensor]` now, which every supported version accepts.

* Define singleton pixel axes at normalized center, reject non-positive normalization sizes, preserve empty warp
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

* Restore the YUV test coverage that #3539 deleted, and fix the `kornia.color` YUV docstring examples (#4045).
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

* Fix `SIFTDescriptor.gk`, `PatchDominantGradientOrientation.weighting` and `PatchAffineShapeEstimator.weighting`
  being plain attributes, so `nn.Module.to()` left them behind on the original device and dtype (#4069). They are
  now non-persistent buffers: `.to()`, `.cuda()`, `.half()` and friends move them, and because they are fully
  determined by `patch_size` they stay out of `state_dict()`, so existing checkpoints still load with
  `strict=True`. The three `forward` methods also cast into a local instead of rebinding the attribute, so a call
  no longer leaves the module's kernel in whatever dtype and device the last input happened to have. Numerical
  output is unchanged: the kernels were already cast to the input's dtype and device inside `forward`.
* Fix patch extraction on MPS returning darkened patches for any LAF that touches an image border (#4063).
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
* Fix `match_fginn`'s geometric consistency check comparing every query's candidates against **query 0**'s
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
* Fix `DenseSIFTDescriptor` registering a process-global cached tensor as its `_poolingconv_weight` buffer, so that
  one instance's `load_state_dict` silently corrupted every other instance and every instance constructed later
  (#4068). `_get_reshape_kernel` memoises `torch.eye(numel)` per `numel` and used to return a *view* of the cached
  tensor; `.float()` on an already-float tensor returns the same object, so no copy happened between the cache and
  the buffer, and `load_state_dict` copies into buffers in place. Loading a checkpoint into one `DenseSIFTDescriptor`
  therefore overwrote the shared identity matrix, with no error raised and every descriptor built from it wrong from
  then on. The cache is removed rather than made safe: cloning on the way out is what a safe cache requires, and a
  clone costs more than rebuilding the identity at every size measured (2.6 us against 2.0 us at the default
  `numel = 8 * 4 * 4`; 3.9 ms against 2.8 ms at the former 4096 bound, which also retained 67 MB for the lifetime of
  the process). Descriptor output is byte-identical to before.
* Fix `laf_is_inside_image` treating the image extent as `(w, h)` rather than `(w - 1, h - 1)`, which made its
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
* Fix the empty-result paths in `kornia.feature.integrated` disagreeing with the corresponding non-empty ones
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

* Stop `MultiResolutionDetector` (and therefore `KeyNetDetector`) fabricating detections, honour its `mask` argument,
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
  absolute `score_threshold`). A floating-point mask is used as weights on the detection scores. The mask must be
  `(1 or B, 1, H, W)` with the image's spatial size. It is resampled onto every pyramid level or octave with a
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
  filters zero LAFs before matching; `match_snn`, `match_mnn` and `match_smnn` reject the block on their own, and
  `LocalFeature.forward` documents the one-line filter for a hand-rolled `match_nn` pipeline. The feature benchmark
  does the same before affine/orientation/description, avoiding meaningless descriptor work and the quadratic
  distance-matrix cost of padded rows; its CUDA timers also synchronize around the measured detector and full
  matching regions.

* `SIFTDescriptor`, `DenseSIFTDescriptor`, `HardNet` and `HardNet8` return NaN for a constant patch in `float16`,
  and the SIFT descriptors' backward is NaN in `float16` wherever two neighbouring pixels are equal.

  `F.normalize`'s default `eps` of 1e-12 is not representable in `float16` and rounds to zero, so the
  `norm.clamp_min(eps)` that exists to stop a zero-norm input becoming `0 / 0` was itself zero in exactly that
  dtype: an all-zero patch normalised to NaN, while `bfloat16`, `float32` and `float64` were fine. A detector that
  pads a short result hands the descriptor a zero LAF, which samples one image point repeatedly and produces
  exactly that patch -- `ScaleSpaceDetector(400)` on a random `64x64` `float16` image gave 365 NaN descriptor rows
  -- and a NaN descriptor is worse than a meaningless one, because it propagates through `torch.cdist` and poisons
  the whole matching. The nine `F.normalize` calls in those four modules now pass an `eps` representable in the
  input dtype. `bfloat16`, `float32` and `float64` results are unchanged: they keep the 1e-12 default.

  The forward pass was only half of it. `SIFTDescriptor` and `DenseSIFTDescriptor` guard `sqrt(gx^2 + gy^2 + eps)`
  and `atan2(gy, gx + eps)` with `eps = 1e-10`, which is zero in `float16` -- and a squared `float16` gradient
  underflows long before that -- so at every pixel with a zero gradient both sat on their singular point and the
  input gradient came back NaN (9 of 4096 on ordinary random `32x32` patches). The gradient magnitude and
  orientation are now computed in float32 for half-precision input and cast back, and the RootSIFT `sqrt` uses the
  same dtype-aware `eps`; wider dtypes are byte-identical.

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
