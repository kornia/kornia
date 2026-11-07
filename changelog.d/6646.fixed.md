`kornia.enhance.adjust_contrast_with_mean_subtraction` (and `AdjustContrastWithMeanSubtraction`) accept a 2-D `(H, W)` image, as documented, and blend it with its mean. It used to raise `IndexError`.
