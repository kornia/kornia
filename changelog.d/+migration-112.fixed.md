`normalize_laf`, `denormalize_laf` and `generate_patch_grid_from_normalized_LAF` count a singleton image axis as
one pixel of extent instead of dividing by `size - 1 == 0`. `normalize_laf` raised `ZeroDivisionError` for a
1-pixel-wide or 1-pixel-tall image -- and with it both patch extractors on their default
`normalize_lafs_before_extraction=True` path -- while `denormalize_laf` silently collapsed every LAF to zero.
The conversions are now finite and round-trip, and both extractors return finite patches for such an image.
