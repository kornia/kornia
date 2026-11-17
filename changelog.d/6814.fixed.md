`transform_points` now gives the same finite Euclidean points for nonzero scalar multiples of a projective matrix
whose homogeneous products remain representable, instead of returning an undivided numerator when the matrix scale
made `w` smaller than `1e-8`.
