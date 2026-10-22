In the auto augmentation policies, `ShearX` / `ShearY` now map the magnitude to degrees with
`rad2deg(atan(magnitude))` — matching the AutoAugment, RandAugment and TrivialAugment papers and
torchvision — instead of multiplying it by `180`, which turned the top-of-range shear factor `0.3`
into `54` degrees, a shear of `tan(54°) ≈ 1.38`. (#6250)
