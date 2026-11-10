`CornerHarris` now keeps a number `k` as a Python float and builds it at each call in the image's dtype, or in float32
for a half-precision image, and `SIFTDescriptor` and `PatchAffineShapeEstimator` build their Gaussian windows in
float64 for a float64 patch. They used to apply `k` and the windows as built once in float32, so float64 results
carried float32 rounding: `CornerHarris(0.04)` used `k = 0.03999999910593033`. A `gk` or `weighting` buffer changed by
hand is therefore no longer used for a float64 patch, and a `CornerHarris` moved with `.half()` or `.bfloat16()` no
longer rounds `k` to that dtype. An older `state_dict` that holds `k` still loads; a saved `k` that differs from the
module's replaces it, as before. Other float32 and half-precision results are unchanged. The pooling weights of
`SIFTDescriptor`, an `nn.Conv2d` parameter, are still built in float32.
