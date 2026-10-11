`equalize`, `equalize3d`, `posterize` and `sharpness` now accept valid empty batches,
including empty leading dimensions, instead of rejecting them or using an ambiguous
reshape. Their corresponding augmentations also preserve empty input shapes, dtype,
device and existing autograd behavior. Empty channel or spatial dimensions, invalid
parameter shapes and images smaller than the sharpness kernel remain unsupported.
Non-empty eager computations and parameter interpretation are unchanged. Part of #4429.
