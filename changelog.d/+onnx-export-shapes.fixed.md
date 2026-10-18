`EdgeDetector.to_onnx` with the pre- and post-processor exports a graph whose edge maps have the input's size, as
eager `forward` returns them. The graph returned the maps at the pre-processor's size (352 x 352 for the default
builder) whatever the input, and only warned that `ResizePostProcessor` is not exported; it now returns one
`(B, 1, H, W)` tensor, the `torch.cat` of the eager per-image list. `image_size` is documented as the graph's input
and output size: the pre-processor keeps the size it was built with. Eager `forward` and `ResizePostProcessor` are
unchanged. (#6120)
