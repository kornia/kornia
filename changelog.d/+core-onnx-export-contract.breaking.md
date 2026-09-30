`ONNXExportMixin.to_onnx` (behind `ImageModule.to_onnx`, `kornia.core.ImageSequential.to_onnx` and the model
exporters) now requests ONNX opset 18, the opset the dynamo exporter builds natively, and documents it. It used to
request 17: the dynamo exporter, the default of `torch.onnx.export` from torch 2.9, met that only when every operator
had a down-conversion adapter (a convolution came back at 17, a `GaussianBlur2d` pipeline at 18 after a logged
conversion failure), and the legacy exporter of older torch returned 17. Pass `opset_version=` to request another one.
The default input shape (`ONNX_DEFAULT_INPUTSHAPE`) is now `[-1, 3, -1, -1]` instead of `[-1, -1, -1, -1]`, so a call
without `input_shape` exports a fixed 3-channel input with a dynamic batch, height and width: on torch 2.9 and later
the old default failed for any model with a fixed channel count, and on older torch it declared the channel dimension
dynamic even where the model fixed it. Pass `input_shape` for another channel count or a dynamic one.
`ONNXSequential(..., auto_ir_version_conversion=True)` now converts the models to the highest opset among them instead
of to 17, so it accepts kornia's own opset-18 exports; on some of them, such as a `GaussianBlur2d` pipeline, it used
to fail with `No Adapter To Version $17 for Pad`.
