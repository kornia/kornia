`SIFTFeature` now detects bright blobs as well as dark ones. Its detector, `MultiResolutionDetector`, keeps the maxima
of its response only, and `SIFTFeature` gave it the single-scale DoG of `BlobDoGSingle`, whose maxima lie at dark
blobs: a bright blob was never detected, only weaker maxima on the ring around it. `SIFTFeature` now gives the
detector the magnitude of the DoG, so a keypoint is a maximum or a minimum of the DoG, as in the DoG detectors of
Lowe, OpenCV and VLFeat, and it must exceed the magnitude of every other response in its suppression window, of either
sign. `SIFTFeature` therefore returns different keypoints, many of them at bright blobs, and its responses are the
magnitude of the DoG; an image and its negative give keypoints at the same positions and scales.
`MultiResolutionDetector` itself and the Key.Net features are unchanged.
