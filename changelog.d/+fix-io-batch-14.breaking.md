`load_image(path, ImageLoadType.UNCHANGED)` returns a grayscale JPEG as one channel, `(1, H, W)`; it used to return
three equal channels, and image modules given a file path, such as `AugmentationSequential` and `ImageSequential`,
now receive `(1, 1, H, W)` for such a file instead of `(1, 3, H, W)`. Invalid arguments now raise a kornia error up
front instead of failing deep inside: `load_image` raises `BaseError` when `desired_type` is not an `ImageLoadType`
(it raised `NotImplementedError`), and `write_image` raises `TypeCheckError` for an input that is not a tensor (it
raised `AttributeError`), `BaseError` for a `quality` that is not an integer in [0, 100] (101 was silently treated as
100, -1 raised `OverflowError` and 80.5 `TypeError`), `BaseError` naming the shape for a batched `(B, C, H, W)`
tensor (kornia-rs raised `TypeError`), and `BaseError` naming the supported channel counts for an image with other
than 1 or 3 channels, or 4 for PNG (kornia-rs raised `ValueError: expected 3 channels`). `BaseError` is not a
`ValueError` or `TypeError`, so code that caught those must catch `kornia.core.exceptions.BaseError`. Writing into a
directory that does not exist raises `FileNotFoundError` naming the directory instead of a bare `Exception`.
`get_sample_images` caches downloads under `<kornia_config.hub_cache_dir>/images` by default instead of a hard-coded
`.kornia_hub/images`; the two are the same path unless `hub_cache_dir` is changed.
