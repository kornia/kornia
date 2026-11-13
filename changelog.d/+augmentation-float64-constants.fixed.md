`RandomFisheye`, `RandomChannelDropout` and the augmentation `Normalize` and `Denormalize` now apply their sampling
grid, `fill_value`, and a `mean` or `std` given as numbers to a float64 image exactly. They used values built in
float32, so float64 results carried float32 rounding: `RandomChannelDropout(fill_value=0.37)` filled the dropped
channels with `0.3700000047683716`. The `fill_value` buffer and the `mean` and `std` flags still hold the float32
values, so a value assigned or changed in place after construction still applies, in float64 too. Float32 and
half-precision results are unchanged, and a `mean` or `std` given as a tensor is used as before.
