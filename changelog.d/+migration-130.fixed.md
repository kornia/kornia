Fix the empty-result paths in `kornia.feature.integrated` disagreeing with the corresponding non-empty ones
(#4065). `get_laf_descriptors` returned a hardcoded width of 128 with the *LAF's* dtype and device when it was
handed no keypoints, so an empty batch through, say, a 256-d descriptor produced a `(B, 0, 128)` result that could
not be concatenated with a real one; it now runs the descriptor once on a zero patch and returns its actual width
(the flattened per-patch output, matching the non-empty path's `.view(B, N, -1)`), dtype and device. That is a
widening of the empty path's failure surface: a third-party descriptor that raises on a zero input now raises here
where it previously returned quietly. `LocalFeatureMatcher.no_match_output` returned `lafs0`/`lafs1` of shape
`(0, 0, 2, 3)` where the success path returns `(1, NC, 2, 3)`, so `lafs0[0]` raised `IndexError` exactly when
nothing matched; the empty tensors now keep the leading batch of 1. That is a shape change on a public return
value -- code using `out["lafs0"].shape[0] == 0` as its no-match signal now sees `1` and should test
`out["keypoints0"].shape[0] == 0`, which was already the consistent signal. `LocalFeatureMatcher.forward`'s
docstring also stops claiming `confidence` is in `[0, 1]`: it is `1 - distance`, which for the raw-distance
matchers `nn` and `mnn` goes at least down to -1 on unit-norm descriptors, and is unbounded below for arbitrary
ones.
