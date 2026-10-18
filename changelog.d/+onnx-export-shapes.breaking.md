An `EdgeDetector` whose `ResizePostProcessor` resizes with `interpolation_mode="area"` now raises a `ValueError` in
`to_onnx` with the pre- and post-processor, and a direct `torch.onnx.export(detector, ...)` fails with it (inside the
exporter's own error on the dynamo exporter). It used to export a graph that returned the maps at the
pre-processor's size. `"area"` resizing is adaptive average pooling, which cannot be exported to the input's size:
build the post-processor with `"bilinear"`, `"bicubic"`, `"nearest"` or `"nearest-exact"` to export the detector.
Eager `forward` with `"area"` is unchanged. (#6120)
