`tversky_loss` and `TverskyLoss` now accumulate spatial sums in float32 for float16 and bfloat16 inputs, preventing overflow on large images while preserving the returned loss dtype.
