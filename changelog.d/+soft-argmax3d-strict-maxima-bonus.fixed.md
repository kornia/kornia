`conv_soft_argmax3d` with `strict_maxima_bonus > 0` now scales a strict maximum by `(1 + strict_maxima_bonus)`, as
documented, and applies it at the right depth level when the depth padding is 0. It used the maximum's value as the
mask, which scaled by `(1 + bonus * value)` and flipped the sign of negative maxima. (#5018)
