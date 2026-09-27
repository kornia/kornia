`spatial_softmax2d`, `spatial_soft_argmax2d` and `SpatialSoftArgmax2d` now treat `temperature` as a softmax
temperature, as `conv_soft_argmax2d` and `conv_soft_argmax3d` already did: the input is divided by it, so a smaller
value gives a sharper distribution. Previously the input was multiplied by `temperature`, so a larger value was
sharper; a Python float raised `AttributeError`; and `0` or a negative value was accepted (`0` returned a uniform map,
a negative value inverted it). A float is now accepted, and a temperature that is not positive raises `ValueError`.
To keep an old result, pass `1 / temperature`. The default, `1.0`, gives the same output as before.
