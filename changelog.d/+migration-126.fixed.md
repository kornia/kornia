Fix patch extraction on MPS returning darkened patches for any LAF that touches an image border (#4063).
MPS has no `padding_mode="border"` for `torch.nn.functional.grid_sample`, so `extract_patches_simple` and
`extract_patches_from_pyramid` emulate it with zero padding and a clamped grid. The clamp was `grid.clamp(-1, 1)`,
but with `align_corners=False` normalized coordinate `±1` is the outer *edge* of the border pixel, not its center,
so bilinear sampling blended the border pixel with the zero padding and returned roughly half its value. The grid
is now clamped to the outermost pixel centers, `±(1 - 1/size)` per axis, which reproduces `padding_mode="border"`
exactly. On a patch overlapping the image corner the maximum deviation from the CPU result drops from `0.395` to
`2.5e-6` on a `[0, 1]` image, with 64.7% of the patch's pixels previously off by more than `1e-3`. Every descriptor
built on these patches — `get_laf_descriptors`, `LAFDescriptor`, `SIFTFeature`, `KeyNetAffNetHardNet` and the rest —
moves with them for keypoints near the border. The CPU and CUDA paths never took this branch and are byte-identical
to before.
