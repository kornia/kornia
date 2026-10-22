An empty batch through a shape-changing augmentation with `p < 1` now comes back at the configured output size, as
with `p=1`, instead of at the input size: `RandomCrop((4, 4), p=0.5)` on a `(0, 3, 6, 8)` batch gives
`(0, 3, 4, 4)`, as in 0.8.3. This covers `RandomCrop`, `CenterCrop`, `RandomResizedCrop`, `Resize`,
`LongestMaxSize`, `SmallestMaxSize`, `RandomCrop3D` and `CenterCrop3D`, the masks that follow the image, and the
sizes `AugmentationSequential` records for its later stages and `inverse`. A non-empty batch that the gate skips
entirely still keeps its input size (#4429).
