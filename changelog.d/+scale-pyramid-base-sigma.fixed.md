`ScalePyramid` with an `init_sigma` below the assumed input blur (`0.5`, or `1.0` with
`double_image=True`) now builds and labels each octave from the first level's real blur, so the
returned `sigmas` match the levels. Such a pyramid is now the same as one built with `init_sigma`
equal to the input blur. The default `init_sigma=1.6` is unchanged.
