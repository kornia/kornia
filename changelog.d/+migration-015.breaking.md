`MultiResolutionDetector` and `KeyNetDetector` change their output: padded slots now read as a zero response and
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
