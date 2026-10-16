`psnr`, `psnr_loss` and `PSNRLoss` compute half-precision inputs in float32 to avoid spurious infinite PSNR and NaN gradients from MSE underflow or overflow, preserving the promoted output dtype.
