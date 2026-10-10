`distort_points_kannala_brandt` computes its float16 reciprocal in float32 to avoid nonfinite point gradients at small nonzero radii, then casts the reciprocal back before applying the distortion.
