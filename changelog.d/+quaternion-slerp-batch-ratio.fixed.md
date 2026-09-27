`Quaternion.slerp` now reads a `(B,)` tensor ratio as one ratio per quaternion, like `(B, 1)`. With `B == 3` it
scaled the components of each rotation vector instead and returned a wrong result, and other `B > 1` raised. (#4991)
