`depth_from_plane_equation` floors a positive `eps` at float16's smallest subnormal, so the default
`eps=1e-8` still guards a grazing ray in float16 instead of rounding to zero and returning `inf`.
