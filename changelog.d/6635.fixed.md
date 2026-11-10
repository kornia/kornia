`binary_focal_loss_with_logits` and `BinaryFocalLossWithLogits` now return zero loss and logit gradients at ignored pixels even when their logits are NaN or infinite.
