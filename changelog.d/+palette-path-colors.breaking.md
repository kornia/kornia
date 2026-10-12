`ImageModuleMixIn`, `ImageSequential` and `AugmentationSequential` now load palette PNG image paths as their RGB
colors, or RGBA colors when the palette has transparency. Previously these paths produced one-channel palette indices
divided by 255, while the same images passed as PIL objects produced their colors.
