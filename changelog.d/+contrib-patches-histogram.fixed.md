`combine_tensor_patches` and `extract_tensor_patches` now recombine and extract patches exactly:

- An all-zero padding or unpadding tuple such as `(0, 0)` or `(0, 0, 0, 0)` no longer switches off `allow_auto_padding` / `allow_auto_unpadding`: a 5 x 7 image with 2 x 2 windows gave 6 patches and dropped the last row and column, and now gives the same 12 patches as `padding=0`.
- `combine_tensor_patches` divides each covered pixel by its exact overlap count instead of `count + eps`, so a `float64` value of `1000.0` is no longer returned as `999.99999`. Pixels no patch covers are `0`, where `eps=0` gave `nan` and a negative `eps` gave `inf`.
- `combine_tensor_patches` sums integer patches in `int64`, so `16777217` is no longer returned as `16777216`; the overlap mean truncates toward zero. Integer patches now work on MPS without a float32 round trip.
- `combine_tensor_patches` sums `float16` and `bfloat16` patches in `float32`: 289 overlapping values up to 255 overflowed to `inf` in `float16` and were off by 64 in `bfloat16`.
- The `eps` argument of `combine_tensor_patches` and `CombineTensorPatches` is deprecated and has no effect; a value other than the default used to change every recombined pixel. Passing it emits a `DeprecationWarning`.

`histogram_matching` and `interp` now return exact values:

- `interp` returns `fp` exactly at a knot, and `np.interp`'s `5.0` instead of `6.0` left of a repeated first knot (`interp([-1, 0, 0.5], [0, 0, 1], [5, 6, 7])`).
- Matching an image to itself is the identity (209 of 5328 `float32` pixels used to change), `[3, 1, 2]` matched onto `[10, 20, 30]` gives `[30, 10, 20]` instead of `[29.999998, 10, 20]` (which became `29` as `uint8`), and a `float32` result no longer leaves the template's range by one ulp.
- An empty source returns an empty tensor of the same shape instead of raising `IndexError`; an empty template raises a `ValueError`.
