`rgb_to_hls` and `RgbToHls` now export to ONNX with the TorchScript exporter and compile with
`torch.compile(fullgraph=True)`. For an input that does not require grad they used to write the H, L and S channels
with `out=` into views of one preallocated tensor. The TorchScript exporter (the `torch.onnx.export` default before
torch 2.9, and `dynamo=False` from it) did not record those writes, so the exported graph ignored its input and returned
zeros, and `torch.compile(fullgraph=True)` raised on the non-contiguous `out=` tensor. The channels are now computed as
separate tensors and stacked; eager results are bit-identical.
