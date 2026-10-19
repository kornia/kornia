`RANSAC` now returns a one-dimensional `(N,)` boolean inlier mask when no model is found, matching successful
calls. Previously, the no-model mask had shape `(N, 1)` and could not be used to index the input keypoints.
