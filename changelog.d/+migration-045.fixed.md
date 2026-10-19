`depth_from_plane_equation` returns a finite depth for a ray exactly parallel to the plane. The
near-singular guard was `eps * torch.sign(denom)`, and `torch.sign` is zero at zero, so at the exact
singularity the epsilon was multiplied away and the division still ran against zero, returning `inf`.
A grazing ray is not an exotic input: it is every pixel on the horizon of a ground plane. The guard
now picks its sign with a comparison, which has no such hole and keeps the sign the small non-zero
denominators already got. `torch.copysign` reads the same but is not exportable, and this function
is in the documented ONNX export surface, so the comparison form is pinned in
`tests/onnx/test_export_coverage.py`. The finite-depth promise is bounded by the dtype: the default
`eps=1e-8` is below float16 resolution, so half-precision callers must pass a representable `eps`.
(#4280, #4348)
