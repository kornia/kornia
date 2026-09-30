SSIM and SSIM3D now compute half-precision image moments in float32, preventing NaN maps for pixel-range images and small dynamic ranges while preserving output dtype, including under autocast.
