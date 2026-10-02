`GeometricAugmentationBase2D.inverse_masks` now keeps mask-specific sampling overrides local to the call, so a failed inverse no longer changes interpolation for subsequent images.
