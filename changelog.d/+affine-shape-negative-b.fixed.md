Fixed `PatchAffineShapeEstimator` treating a negative off-diagonal moment as degenerate, which gave a patch
and its mirror image different shapes (a thin ellipse versus a circle).
