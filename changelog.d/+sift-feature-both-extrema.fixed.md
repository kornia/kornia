`SIFTFeature` now detects bright blobs as well as dark ones. Its detector, `MultiResolutionDetector`, keeps the maxima
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
`MultiResolutionDetector` itself and the Key.Net features are unchanged.
