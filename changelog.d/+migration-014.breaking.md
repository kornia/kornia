The shape guards of `normalize_homography` and `denormalize_homography` now reject everything but a
`(3, 3)`/`(B, 3, 3)` matrix, and `normalize_homography3d`'s everything but a `(4, 4)`/`(B, 4, 4)` one (#3999).
A rank-4 input used to pass the guard and come back with its rank unchanged —
`normalize_homography(eye(3).expand(2, 4, 3, 3), (4, 5), (8, 9))` returned a `(2, 4, 3, 3)` matrix and now raises
`ValueError`. A wrong-sized input, such as a `(B, 4, 4)` to the 2-D functions, used to pass the guard and fail
later inside `matmul`; it now raises at the guard, and `normalize_homography3d`'s message names `Bx4x4` instead of
`Bx3x3`. Unbatched `(3, 3)`/`(4, 4)` matrices are still accepted and still promoted to a leading batch of 1.

One side effect reaches tracing callers: the rewritten guard evaluates its shape comparison unconditionally, where
the old `or`-form short-circuited past it for a rank-3 input, so `torch.jit.trace` of any of the three functions
now emits a `TracerWarning` about converting a tensor to a Python boolean that it did not emit before. The guard
is a static check and the traced graph is unchanged — the warning is noise, not a correctness signal.
