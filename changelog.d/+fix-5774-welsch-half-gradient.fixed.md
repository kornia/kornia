`welsch_loss` now computes half-precision residuals in float32, preventing NaN gradients when finite inputs have a difference outside the float16 range.
