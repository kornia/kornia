`LocalFeatureMatcher` no longer matches the zero-LAF slots with which a fixed-shape detector pads an under-filled
result, and now forwards its documented `mask0`/`mask1` inputs to feature extraction. `nn` and `mnn` callers can
therefore receive fewer correspondences: identical descriptors sampled at padded origin frames are no longer
reported as matches. Three-dimensional `(B, H, W)` masks keep working and are promoted to the detectors'
`(B, 1, H, W)` form; four-dimensional masks are forwarded unchanged. Because the masks now reach the detector,
they are subject to its check: a mask that is not at the image's spatial size, which used to be ignored, is
rejected. A floating-point mask changes meaning in `ScaleSpaceDetector`-based pipelines too; see the mask semantics entry under **Bug fixes**.
