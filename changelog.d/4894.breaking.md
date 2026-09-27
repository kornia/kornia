A mask override with `align_corners=None` in `AugmentationSequential(extra_args=...)` now means the module's own
`align_corners` in `cropping_mode="slice"` as well as in `"resample"`. Slice mode used to keep `None`, which
`interpolate` reads as `False`, so with `RandomResizedCrop`'s default `align_corners=True` a bilinear or bicubic
mask was resized on a different grid from its image and no longer lined up with it. Outputs change only for that
combination: slice-mode `RandomResizedCrop` with a bilinear or bicubic mask override, `align_corners=None`, and
`align_corners=True` on the module. Nearest masks, `align_corners=False` and resample mode are unchanged, as are
`RandomCrop` and `CenterCrop`. Fixes #4854.
