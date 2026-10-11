`laf_is_inside_image` now tests the exact extent of each region. It used to check ten points sampled on the region's
boundary, so an ellipse that crossed an image edge between two of them, by up to 4.9 % of its extent along that axis,
was reported as inside. It now compares the exact span of the ellipse along each image axis, `c_i ± ‖A[i, :]‖`, with
the same `border` and the same closed `0 .. w - 1` / `0 .. h - 1` bounds. `ScaleSpaceDetector` applies the same rule
to half of each candidate's frame against a 5-pixel octave border, and therefore so do `HesAffNetHardNet`,
`GFTTAffNetHardNet` and `SIFTFeatureScaleSpace` with its default patch backend: a candidate whose half frame overhung
that border along x by less than 4.9 % of its radius used to be kept and is now discarded, and its slot goes to the
next candidate. On the first image of each of the eight Oxford affine sequences, this replaces 10 to 15 of the 16384
detections per pipeline. `SIFTFeature` and the KeyNet pipelines are unchanged.

`LAFAffineShapeEstimator` and `LAFAffNetShapeEstimator` now compose the shape they estimate with the shape of the
input LAF. Both sample each patch through the upright form of the input LAF, so the estimate is in that frame's
coordinates; it used to be written over the input shape as if it were measured in the image, which is right only for
an isotropic input. For an anisotropic or skewed input the output was the correction rather than the corrected shape,
and applying an estimator to its own output alternated between the region's shape and a circle (axis ratios 2.00,
1.01, 2.01, 1.02 on a Gaussian blob of ratio 2). The output region is now that frame composed with the estimate, with
the input's centre and scale and the same `preserve_orientation` behaviour, so an input that already matches the
region keeps its shape and repeated application converges. An isotropic input gets the same result as before. The
frames `ScaleSpaceDetector` hands to its affine module are circles in octave pixels, which the octave-to-image factors
`(W - 1) / (W_o - 1)` and `(H - 1) / (H_o - 1)` stretch by up to 2 % between x and y, so the frames of
`HesAffNetHardNet` and `GFTTAffNetHardNet` move by about 1e-4 of their scale (median), and on the first images of the
eight Oxford affine sequences about 50 of each pipeline's 16384 descriptors change by more than 0.01.
`KeyNetAffNetHardNet` is unchanged.
