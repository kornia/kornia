`RANSAC(score_type="msac")` accepts models again. Acceptance and confidence stopping treated the MSAC score as an inlier
count, and a hypothesis was kept only when that score exceeded the minimal sample size. The old score subtracted the
truncated squared residuals from `N`, so with a 2-pixel threshold it usually fell below that cutoff and the estimator
returned the all-zero failure matrix: eight-point fundamental estimation on 60 IMC phototourism pairs failed for 87% of
SIFT and all XFeat pairs. With thresholds below 1 the score could instead exceed the inlier count and stop sampling too
early. The score is now `sum(1 - min(e / inl_th**2, 1))`, the inlier count is tracked separately for acceptance and
stopping, and those pairs no longer fail. Early stopping is checked after every batch using the support after local
optimization. Zero and nonfinite candidate matrices are rejected, since a zero fundamental matrix has zero Sampson
error for every match. Five- and seven-point estimates are kept even when they have fewer inliers than the eight-point
polisher needs. Local optimization keeps a full-inlier refit that ties the score instead of discarding it,
so RANSAC scoring no longer returns the less precise minimal-sample model when support does not grow. The failure mask has shape `(N,)`, like a success, and the essential-matrix mask is recomputed after
projection onto the essential manifold.
