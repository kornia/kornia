`kornia.geometry.homography.find_homography_dlt` with the default `solver="lu"` and five or more points, and
`find_homography_dlt_iterated`, return the homography instead of an all-NaN matrix on exact correspondences
whose normal matrix has an exactly zero last LU pivot, such as identical point sets, integer shifts and exact
90-degree rotations of integer keypoints (about 1 in 6 random sets of 5 to 30 points). Such a system has rank 8,
so the homography is unique and now matches `solver="svd"` and `cv2.findHomography`. Coincident and collinear
points still give NaN.
