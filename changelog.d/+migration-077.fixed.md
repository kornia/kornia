`ConvQuadInterp3d` / `conv_quad_interp3d` no longer return NaN gradients for `float16` input. The
Hessian determinant `_solve_cramer_sym3x3` divides by is a product of three second derivatives, so for
a `[0, 1]` response it lands around 1e-4 and below and still clears the `eps` gate. The forward divides
by it once and stays finite, but the backward scales by `1 / det**2`, which `float16` cannot represent
(`finfo(float16).tiny` is 6.1e-5), so the gradient overflowed and reduced to NaN over part of the
volume. The solve is now promoted to `float32` for `float16` input and the shifts cast back, as #4231
already does for the half-precision meshgrid. `bfloat16` keeps `float32`'s exponent range and was never
affected; `bfloat16` and `float32` outputs are unchanged. (#4305)
