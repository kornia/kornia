`focal_loss` and `FocalLoss` now exclude ignored logits before log-softmax, preventing extreme finite logits at ignored positions from producing NaN losses and gradients.
