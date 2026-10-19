`SIFTDescriptor`, `DenseSIFTDescriptor`, `HardNet` and `HardNet8` return NaN for a constant patch in `float16`,
and the SIFT descriptors' backward is NaN in `float16` wherever two neighbouring pixels are equal.

`F.normalize`'s default `eps` of 1e-12 is not representable in `float16` and rounds to zero, so the
`norm.clamp_min(eps)` that exists to stop a zero-norm input becoming `0 / 0` was itself zero in exactly that
dtype: an all-zero patch normalised to NaN, while `bfloat16`, `float32` and `float64` were fine. A detector that
pads a short result hands the descriptor a zero LAF, which samples one image point repeatedly and produces
exactly that patch -- `ScaleSpaceDetector(400)` on a random `64x64` `float16` image gave 365 NaN descriptor rows
-- and a NaN descriptor is worse than a meaningless one, because it propagates through `torch.cdist` and poisons
the whole matching. The nine `F.normalize` calls in those four modules now go through one helper that
normalises a `float16` input in float32 and casts back (an `eps` representable in `float16` is safe but not
neutral: a norm in the subnormal window came back as 0.5). `bfloat16`, `float32` and `float64` results are
unchanged: they keep the 1e-12 default.

The forward pass was only half of it. `SIFTDescriptor` and `DenseSIFTDescriptor` guard `sqrt(gx^2 + gy^2 + eps)`
and `atan2(gy, gx + eps)` with `eps = 1e-10`, which is zero in `float16` -- and a squared `float16` gradient
underflows long before that -- so at every pixel with a zero gradient both sat on their singular point and the
input gradient came back NaN (9 of 4096 on ordinary random `32x32` patches). The gradient magnitude and
orientation are now computed in float32 for `float16` input and cast back, and the RootSIFT `sqrt` is computed
in float32 for `float16` as well. An *exactly* flat patch -- what a zero-LAF padding slot samples -- was the
remaining case: the guard gave every zero-gradient pixel a magnitude of `sqrt(eps)`, so a flat patch's
descriptor was a unit vector built from `eps` (float32) or a subnormal one (float16) whose `1 / norm` gradient,
together with `atan2`'s `1 / eps`, overflowed through the float16 cast into an all-NaN input gradient (and was
a meaningless ~1e8 in float32). A zero-gradient pixel now contributes a zero magnitude, a zero vector normalises
to zero with a zero gradient (the `eps` clamp's `1 / eps` was never meaningful), and
`PatchDominantGradientOrientation` skips its parabolic refinement on an empty or uniform histogram instead of
dividing `0 / 0`; a flat patch has a zero SIFT descriptor and a zero input gradient in every dtype. `bfloat16`,
whose exponent range holds both the guard and the squares, and the wider dtypes are byte-identical at every
pixel whose gradient is not exactly zero; a patch with such pixels (a saturated region) loses their `sqrt(eps)`
contribution, a change of at most `1e-5` per pixel before normalisation.
