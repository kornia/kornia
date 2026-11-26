`otsu_threshold` now counts histogram populations in int64 before normalization, preventing an incorrect threshold when a float32 histogram bin contains more than `2**24` pixels.
