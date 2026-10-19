`kornia.contrib.diamond_square` generates fractals with a spatial side below 3 px. A 1 px side raised
`ValueError: math domain error` and a 2 px side a `TypeError` from a float in the `torch.rand` size, so
`RandomPlasmaBrightness`, `RandomPlasmaContrast` and `RandomPlasmaShadow` failed on such images. A small
side is now drawn on the 3 px grid and sliced; outputs for sides of 3 px and more are bit-identical.
(#4472, #4488)
