`validate_bbox` and `validate_bbox3d` flatten rank-4 `(B, N, 4, 2)` / `(B, N, 8, 3)` input with `reshape`
instead of `view`, so a non-contiguous leading-dimension stride (a transpose, a slice that drops boxes, an
`expand`) returns a boolean as documented instead of raising `RuntimeError`. (#4174)
