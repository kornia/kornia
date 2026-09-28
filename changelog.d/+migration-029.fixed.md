`axis_angle_to_rotation_matrix` accepts the `(*, 3)` shape its own guard message promises, instead of
only `(N, 3)`. The body did `wxyz.unbind(dim=1)` and `.view(-1, 3, 3)`, so an unbatched `(3,)` raised
`IndexError: Dimension out of range` and any extra batch dimension raised `ValueError` out of the unbind,
three frames below kornia's own shape guard. Every sibling conversion in the module already accepted
`(*, 3)`, so `axis_angle_to_rotation_matrix(rotation_matrix_to_axis_angle(R))` composed for an `(N, 3, 3)`
rotation matrix and for nothing else, including for the shape `rotation_matrix_to_axis_angle`'s own
doctest returns. This is additive: `(N, 3)` output and gradients are byte-identical, and only shapes that
used to raise now return (closes #3955). (#4342)
