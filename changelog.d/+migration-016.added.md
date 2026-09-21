`kornia.feature.laf_is_filled(laf)` returns the `(B, N)` mask of the slots a detection filled, the complement
of a detector's zero-LAF padding. Fixed-shape detectors pad the slots no detection filled with an all-zero LAF,
and those slots must be dropped before matching -- their descriptors are identical to one another, so they match
each other at zero distance and a mutual nearest-neighbour test does not reject them. `LocalFeature.forward`'s
docstring already required callers of a hand-rolled pipeline to do this and gave them the expression to copy;
they can now call the function instead. It replaces the same expression at four internal sites in
`kornia/feature`. (#4223)
