Eager bbox validation now rejects NaN and infinite coordinates: `validate_bbox` returns `False` and
`Boxes.from_tensor(..., validate_boxes=True)` raises `ValueError` for the `xyxy`, `xyxy_plus` and `xywh`
modes, instead of accepting them as valid geometry (closes #4238). (#4243)
