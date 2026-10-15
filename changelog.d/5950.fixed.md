`kornia.geometry.bbox.transform_bbox` accepts zero boxes and an empty batch. A `(0, 4)` box tensor, or a
`(0, N, 4)` or `(0, N, 4, 2)` batch, raised `RuntimeError: cannot reshape tensor of 0 elements` and now returns an
empty tensor of the input's shape. Non-empty results are unchanged.
