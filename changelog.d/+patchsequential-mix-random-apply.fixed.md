`PatchSequential` with `random_apply` now permits a mix augmentation such as `RandomMixUpV2` to be selected, at most once per forward call. Previously, mix members were never selected.
