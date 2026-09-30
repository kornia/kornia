The unimplemented `Keypoints` and `Keypoints3D` paths now raise `NotImplementedError` with a message naming what
is unsupported instead of a bare one: building either class from a list of tensors,
`to_tensor(as_padded_sequence=True)` on both, and `Keypoints3D.pad`, `unpad`, `transform_keypoints` and
`transform_keypoints_`. `Keypoints3D.transform_keypoints` points at
`kornia.geometry.linalg.transform_points(M, keypoints.data)`, and its `Args` block now documents `M` as
:math:`(4, 4)` or :math:`(B, 4, 4)` rather than the 2D :math:`(3, 3)` shape. The paths remain unimplemented.
