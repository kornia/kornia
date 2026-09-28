`iterative_quad_interp3d`'s `max_candidates` cap is now a per-image budget rather than one shared
across the batch. The `topk` ranked the flattened `(B*C)` candidate list, so an image's refined
keypoints depended on which other images shared its batch: a quiet image next to a high-contrast
one got none. This also reaches `IterativeQuadInterp3d` and `AdaptiveQuadInterp3d` in `patch` mode.
The docstring already read as a per-image budget. A negative `max_candidates` now raises
`ValueError`; it previously raised a `topk` `RuntimeError`, and under the per-image ranking it
would otherwise have silently disabled refinement instead. (#4256, #4350)
