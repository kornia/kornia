`laf_is_inside_image` now tests the exact extent of each region. It used to check ten points sampled on the region's
boundary, so an ellipse that crossed an image edge between two of them, by up to 4.9 % of its extent along that axis,
was reported as inside. It now compares the exact span of the ellipse along each image axis, `c_i ± ‖A[i, :]‖`, with
the same `border` and the same closed `0 .. w - 1` / `0 .. h - 1` bounds. `ScaleSpaceDetector` applies the same rule
to half of each candidate's frame against a 5-pixel octave border, and therefore so do `HesAffNetHardNet`,
`GFTTAffNetHardNet` and `SIFTFeatureScaleSpace` with its default patch backend: a candidate whose half frame overhung
that border along x by less than 4.9 % of its radius used to be kept and is now discarded, and its slot goes to the
next candidate. On the first image of each of the eight Oxford affine sequences, this replaces 10 to 15 of the 16384
detections per pipeline. `SIFTFeature` and the KeyNet pipelines are unchanged.
