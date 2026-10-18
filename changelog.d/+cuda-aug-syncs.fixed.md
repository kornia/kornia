Augmentations no longer synchronize with the GPU to move their parameters in eager mode: parameters sampled on the CPU
and per-call constants reach CUDA inputs without a `cudaStreamSynchronize`, in every 2D, 3D, mix, container and
automatic-policy augmentation, in `forward` and `inverse`, and with mask, box, keypoint and class inputs. Values are
unchanged bit for bit, and compiled graphs keep their copies. The synchronizations that remain are the ones whose
result the host needs: `inverse()` and the 3D warps invert the transformation matrix with `torch.linalg.inv`, which
raises for a singular matrix; box inputs are validated (now in one host read instead of three); and
`RandomTransplantation` and `AdaptiveDiscriminatorAugmentation` draw from data on the device. Parameters as large as
the image (noise, plasma and illumination maps, elastic noise, salt-and-pepper masks) still cost the host the time of a
pageable copy, and an augmentation moved to the GPU with `.to("cuda")` samples its parameters on the GPU and still
reads them back.
