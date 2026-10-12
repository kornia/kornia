`PinholeCamera.scale` no longer shares its `extrinsics` tensor with the source camera, so
`scaled.tx = 7` leaves the camera `scale()` was called on unchanged. The returned camera's numbers are unchanged. (#4264, #4349)
