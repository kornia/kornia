`DescriptorMatcherWithSteerer(normalize=True)`, `DiscreteSteerer.steer_descriptions(normalize=True)` and the
`MKD` descriptors, like `SIFTDescriptor` and `HardNet`, L2-normalise a float16 input in float32 and cast back,
so an all-zero float16 descriptor normalises to zero instead of NaN (the 1e-12 guard underflows in float16) and a
descriptor whose norm sits below the smallest float16 normal still normalises to one rather than being clamped.
`PatchDominantGradientOrientation`, and with it `LAFOrienter`, take the gradient magnitude and orientation of a
float16 patch through the same float32 step `SIFTDescriptor` uses, so a flat patch yields a finite angle instead
of a NaN LAF -- which reached a *filled* detection slot in a float16 `SIFTFeature` pipeline and poisoned that
descriptor row. `SIFTDescriptor(rootsift=True)` and `DenseSIFTDescriptor(rootsift=True)`
compute the RootSIFT square root in float32 for a float16 input: the float16-representable guard would have
read every empty bin as `sqrt(6.1e-5)` and biased the descriptor norm to ~1.004. Float32 and float64 outputs are
unchanged.
