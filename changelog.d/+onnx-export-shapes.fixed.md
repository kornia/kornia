`EdgeDetector.to_onnx` with the pre- and post-processor exports a graph whose edge maps have the input's size, as
eager `forward` returns them. The graph returned the maps at the pre-processor's size (352 x 352 for the default
builder) whatever the input, and only warned that `ResizePostProcessor` is not exported; it now returns one
`(B, 1, H, W)` tensor, the `torch.cat` of the eager per-image list. `image_size` is documented as the graph's input
and output size: the pre-processor keeps the size it was built with. Eager `forward` and `ResizePostProcessor` are
unchanged. (#6120)
`Sam.to_onnx` of a MobileSAM (TinyViT image encoder) on the TorchScript exporter, the default before torch 2.9 or
`dynamo=False`, exports a graph that runs at the batch of `pseudo_shape` (1 by default) and declares it. Since
dynamic dimensions are traced at 2 or more, that graph declared a dynamic batch but ran at batch 2 only, so a
batch-1 call with the default `pseudo_shape` failed in ONNX Runtime. Other encoders, and the dynamo exporter, keep
a dynamic batch. (#6125)
