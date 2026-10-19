`infer_bbox_shape` and `bbox_to_mask` now reject rank-4 `(B, N, 4, 2)` inputs with `ShapeError`. Previously,
`N < 3` raised an incidental `IndexError`, while `N >= 3` silently returned `(B, 2)` extents computed from the
wrong vertices. (#4218)
