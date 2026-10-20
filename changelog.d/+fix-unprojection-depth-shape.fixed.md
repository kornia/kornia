`unproject_points` and `PinholeCamera.unproject` now reject depth tensors whose leading shape does not match the input points instead of silently broadcasting incorrect depths.
