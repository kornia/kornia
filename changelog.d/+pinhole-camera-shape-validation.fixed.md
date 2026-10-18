`PinholeCamera` now rejects rank-4 intrinsics and extrinsics at construction with a clear shape error. Use `PinholeCamerasList` for the `(B, N, 4, 4)` camera-list layout.
