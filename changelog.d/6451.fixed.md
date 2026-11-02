`PatchDominantGradientOrientation` now gives zero smoothing-weight gradients for flat patches, preventing NaN parameter gradients when a batch includes a flat patch.
