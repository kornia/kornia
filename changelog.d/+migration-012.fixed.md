`RandomMosaic` now preserves `(H, W)` for non-square inputs when `output_size=None`; it previously
returned `(W, H)`. `start_ratio_range` now scales x by width and y by height; it previously scaled
x by height and y by width. (#4459)
