Reset an automatic augmentation policy's cached transformation matrix before applying it again, both for a
sub-policy inside `TrivialAugment`, `RandAugment` and `AutoAugment` and for those policies nested inside
`AugmentationSequential`. Consecutive calls that selected the same geometric policy could transform the input with
the latest parameters while exposing the previous call's matrix through `transform_matrix`, the sub-policy
accumulated one matrix per call for the lifetime of the module, and a nested policy never recorded its parameters
or matrix at all. `transform_matrix` on a policy that has not run yet now returns `None` instead of raising
`AttributeError` (#4171).
