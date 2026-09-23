`TrivialAugment`, and `PolicySequential` when used directly, now sample each operation through its
`OperationBase` wrapper, so a policy entry means what it means under `RandAugment`. They called the
wrapped augmentation's own sampler, which skipped the wrapper's magnitude mapping: `("shear_x", -0.3, 0.3)`
sheared by at most `0.3` degrees instead of up to `54`, and the symmetric ops (`rotate`, `shear_x`,
`shear_y`, `translate_x`, `translate_y`) never drew a negative magnitude. A `PolicySequential` built from
operations with an `initial_magnitude` now uses that magnitude, as the wrapper does. `AutoAugment` samples
as before: its shear bins were pre-scaled to degrees to work around the skipped mapping and are now
fractions, which the mapping scales once. Seeded `TrivialAugment` pipelines draw different magnitudes.
Fixes #4441.
