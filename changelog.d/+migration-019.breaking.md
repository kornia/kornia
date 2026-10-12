`ellipse_to_laf` no longer raises on a degenerate ellipse. One whose `a` or `c` is `0` after rounding to the input
dtype makes the matrix being inverted singular; the batched `torch.inverse` it now replaces (see **Bug fixes**) failed the
whole call with `linalg.LinAlgError`, and the closed form returns non-finite values in that LAF instead, leaving
the other rows of the batch usable. Callers that relied on the exception to reject bad input -- reading
Oxford-format `.ellipse` files, say -- should test the result with `kornia.feature.laf_is_valid`, since an invalid
LAF otherwise propagates silently through `get_laf_scale` and into matching. Detecting the degenerate rows up front
would cost a device synchronisation and the `fullgraph=True` capture, so it is left to the caller. The in-repo
affine estimators handle invalid candidates before nonlinear LAF operations. `PatchAffineShapeEstimator` computes
half-precision gradients, Gaussian weights, moments, and normalization in `float32`, preserving valid anisotropic
shapes that would otherwise underflow; `LAFAffineShapeEstimator` sanitizes non-positive-definite ellipse
coefficients before conversion; and both it and `LAFAffNetShapeEstimator` use a finite fallback with finite
backward. The fallback is the input LAF when `preserve_orientation=True` and its upright form otherwise; a finite
zero-scale input keeps its center and zero affine block.
