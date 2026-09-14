Corrected the matrix-shape checks shared by `PinholeCamera` and `PinholeCamerasList`, the
`pixel2cam` coordinate-shape check, and the `cam2pixel` coordinate/projection checks. Invalid
inputs now raise the intended `ValueError` instead of being accepted or failing later in tensor
operations. The rank-4 matrices used by `PinholeCamerasList` remain supported. (#4387)
