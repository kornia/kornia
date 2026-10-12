`unproject_meshgrid` now requires camera intrinsics of shape `(B, 3, 3)`, rejecting extra
camera axes before they can broadcast across pixel columns and reporting the caller's original shape. (#4383)
