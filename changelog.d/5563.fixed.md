`focal_loss` no longer returns NaN gradients when a softmax probability rounds to 0 or 1 with a fractional focusing parameter (`0 < gamma < 1`).
