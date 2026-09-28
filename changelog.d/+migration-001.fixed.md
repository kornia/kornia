Made uncompiled `RandomGaussianIllumination` instances serializable with `pickle` and `torch.save`,
preserving parameter replay and the `compile()` execution path after restoring the module. (#4457)
