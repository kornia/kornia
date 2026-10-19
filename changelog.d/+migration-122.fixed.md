Fix `torch.jit.script` of `rgb_to_yuv`, `yuv_to_rgb`, `rgb_to_xyz` and `xyz_to_rgb` on older PyTorch (#4043). Their
shared `kornia.color.utils._apply_linear_transformation` helper annotated its optional argument with the PEP 604
`torch.Tensor | None` form, which the TorchScript compiler on the declared floor does not accept (reproduced on
PyTorch 2.1.2, while PyTorch 2.9.1 compiles it), so scripting any of the four conversions failed. The annotation is
`Optional[torch.Tensor]` now, which every supported version accepts.
