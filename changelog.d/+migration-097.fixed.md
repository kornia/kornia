`find_homography_dlt(..., solver="lu")` now solves the minimal four-point system with an adaptive
homogeneous gauge instead of applying LU to its singular normal matrix. This is the estimator
`RANSAC(model_type="homography")` calls for every minimal sample, so homography RANSAC results
change on every platform, not only where the old formulation broke down: over 2048 minimal
samples the median transfer residual drops from 2.3e-5 to 1.9e-6 and the worst case from 4.2e+5
to 4.5e+2. It also fixes the all-NaN homographies seen for a four-point sample on Windows with
PyTorch 2.14, and removes the corresponding CPU half-precision known failures. A four-point
sample containing a zero weight now returns a finite homography instead of NaN. Samples of five
or more points keep the previous normal-equation formulation and are numerically unchanged.
(#4197)
