`CenterCrop` and `RandomCrop` raise a `ValueError` for a negative `size` in the default `cropping_mode="slice"`, as
they already did with `cropping_mode="resample"`. Slice mode used to return an empty crop from `CenterCrop`, and to
fail in `RandomCrop` with an unrelated error about the input image size or a negative tensor dimension.
