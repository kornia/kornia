`infer_bbox_shape3d` and `bbox_to_mask3d` reject rank-4 `(B, N, 8, 3)` input with a `ShapeError`,
the way `infer_bbox_shape` and `bbox_to_mask` have since #4218. `validate_bbox3d` accepts the rank-4
form and reshapes internally, but both callers index dim 1 as the vertex axis, so the box axis was
read as the vertices: an out-of-bounds error with one box, and three `(1, 3)` tensors -- one value
per coordinate rather than per box -- with eight. (#4248, #4351)
