`PinholeCamera.scale` no longer shares its `extrinsics` tensor with the source camera. The
intrinsics were cloned, the extrinsics were handed to the new object by reference, and the
constructor stores what it is given, so `scaled.tx = 7` moved the camera `scale()` was called on.
The returned camera's numbers are unchanged. (#4264, #4349)
