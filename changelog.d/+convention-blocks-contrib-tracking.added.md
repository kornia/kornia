The API documentation of `kornia.tracking.HomographyTracker` and of the numeric and tooling exports of `kornia.contrib`
(patch extraction and recombination, `histogram_matching`, `interp`, `diamond_square`, both connected-component
functions, `distance_transform`, `KMeans`, `Lambda`, `ImageStitcher` and the augmentation audit) states their
conventions, among them the patch order and padding layout, `interp`'s extrapolation where `np.interp` clamps, the
direction of the distance transform against SciPy and OpenCV, the connectivity of the labeling functions and the
direction of the tracker's homography. The contrib page lists `histogram_matching` and `interp`, gives
`diamond_square` its own section and maps the connectivity onto SciPy, OpenCV and scikit-image. Known defects are
documented with their issues: #5956, #5959, #5963, #5967, #5968 and #5971.
