`solve_quartic` returns finite gradients for a pure biquadratic such as `x^4 - 16`. Its two
`torch.clamp(..., min=0.0).sqrt()` sites do not guard the gradient they look like they guard:
`d(sqrt)/dx` is unbounded at 0, and on torch below 2.14 `clamp` passes the incoming gradient
through at the bound rather than zeroing it, so on the older half of kornia's supported torch
range the backward returned `inf` and then `nan` (#4229). Both sites now substitute a safe
radicand under the `sqrt`, as `solve_quadratic` in the same module already did. Forward values
are unchanged. (#4339)
