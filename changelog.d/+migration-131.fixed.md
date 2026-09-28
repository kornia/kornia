Stop `MultiResolutionDetector` (and therefore `KeyNetDetector`) fabricating detections, honour its `mask` argument,
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
