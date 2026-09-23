`spatial_gradient(..., order=2, normalized=True)` now scales every second order kernel by its response to the
quadratic it estimates, so `dxx`, `dxy` and `dyy` come out in the same units: on `x**2 / 2`, `x * y` and
`y**2 / 2` each channel returns 1 in magnitude in both modes. Previously each kernel was divided by its own
absolute sum, which gave the mixed channel a scale of its own: `mode="sobel"` returned `dxy` too large by
`64 / 36`, `mode="diff"` returned `dxx` and `dyy` too small by 4. `order=1` and `normalized=False` are
unchanged.

`hessian_response` and `BlobHessian` use the normalized default, so their values change: the determinant is
now `fxx * fyy - fxy**2` instead of `fxx * fyy - (64 / 36)**2 * fxy**2` (sobel) or `fxx * fyy / 4 - fxy**2`
(diff). At the centre of a 9x9 grid:

| surface (true determinant) | sobel before | sobel after | diff before | diff after |
|---|---|---|---|---|
| `x**2 + y**2` (4) | 4.0 | 4.0 | 0.25 | 4.0 |
| `x * y` (-1) | -3.1605 | -1.0 | -1.0 | -1.0 |
| `(x + y)**2 / 2` (0) | -2.1605 | 0.0 | -0.9375 | 0.0 |

Blob maxima keep their value, saddle responses shrink, so the ranking of `ScaleSpaceDetector` and
`MultiResolutionDetector` with `BlobHessian` can change between blob and saddle candidates. Refs #4710.
