Augmentation classes no longer exhaust a shared `torch.compile` cache. Compiled
`RandomResizedCrop` slice mode keeps sampled crop coordinates in tensors, avoiding
recompilation as crop boxes change while preserving resize interpolation and input gradients.
