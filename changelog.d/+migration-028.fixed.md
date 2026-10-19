`Boxes3D.from_tensor(..., validate_boxes=True)` rejects non-finite coordinates, and `validate_bbox3d`
returns `False` for them instead of raising an `AssertionError` that names the wrong defect. This is the
3D counterpart of #4243. An `inf` passed the positive-extent checks outright, and a `NaN` passed them
because every comparison against `NaN` is `False`, so the box was constructed with non-finite vertices;
in `validate_bbox3d` the `NaN` instead reached the `allclose` extent comparisons and raised
"Boxes must have be cube, while get different widths". `validate_bbox3d`'s four internal callers use it
for its raise and discard the result, so they now convert the `False` themselves and keep raising the same
`AssertionError` they did before, with a message that now names the non-finite coordinates instead of
reporting mismatched cube extents. The `validate_boxes=False` opt-out and the export gate are unchanged
(closes #4258). (#4343)
