float16 and bfloat16 geometry warps now run in float32 and return the input dtype on every device, not only on CPU
(#5814): `warp_affine`, `warp_perspective`, `remap` and the 3D warps build their sampling grid and sample in float32,
and `HomographyWarper`, `elastic_transform2d`, `warp_image_tps` and the depth warps sample in float32. On CUDA and MPS
they used to compute in the half dtype: a bfloat16 `warp_affine` of a 256 px image was off by up to 0.9 on a `[0, 1]`
image on CUDA and 0.72 on MPS, `warp_affine3d` raised for a half CUDA output with a size-1 axis, and a bfloat16 warp
inside a float16 `torch.autocast` raised.
