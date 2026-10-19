Fix `SIFTDescriptor.gk`, `PatchDominantGradientOrientation.weighting` and `PatchAffineShapeEstimator.weighting`
being plain attributes, so `nn.Module.to()` left them behind on the original device and dtype (#4069). They are
now non-persistent buffers: `.to()`, `.cuda()`, `.half()` and friends move them, and because they are fully
determined by `patch_size` they stay out of `state_dict()`, so existing checkpoints still load with
`strict=True`. The three `forward` methods also cast into a local instead of rebinding the attribute, so a call
no longer leaves the module's kernel in whatever dtype and device the last input happened to have. Numerical
output is unchanged: the kernels were already cast to the input's dtype and device inside `forward`.
