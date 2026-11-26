`dice_loss` and `DiceLoss` now ignore non-finite logits at `ignore_index` pixels in both the standard and optimized CPU implementations, preventing NaN loss and gradients.
