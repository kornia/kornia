`normalize_quaternion` floors a positive `eps` at the smallest positive `float16` subnormal, so its default
`eps=1e-12` remains an active division guard and a zero quaternion returns finite zeros instead of NaNs without
changing non-zero representable `float16` magnitudes. An explicit `eps=0.0` still disables the guard. This also
keeps `Quaternion.normalize`, `quaternion_to_rotation_matrix`, and their downstream pose conversions finite for
that input, matching the existing `float32`, `float64`, and `bfloat16` behavior (#4162).
