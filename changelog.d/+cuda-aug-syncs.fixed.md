Augmentations no longer wait for queued CUDA work: parameters sampled on the CPU and per-call constants reach the
GPU without a host synchronization, so every 2D, 3D, mix, container and automatic-policy augmentation, in `forward`
and `inverse` and with mask, box and keypoint inputs, now queues its kernels without stalling the host. Values are
unchanged bit for bit. The synchronizations that remain are the ones whose result the host needs: `inverse()` and
the 3D warps invert the transformation matrix with `torch.linalg.inv`, which raises for a singular matrix; box inputs
are validated (now in one host read instead of three); and `RandomTransplantation` and
`AdaptiveDiscriminatorAugmentation` draw from data on the device. An augmentation moved to the GPU with `.to("cuda")`
samples its parameters on the GPU and still reads them back.
