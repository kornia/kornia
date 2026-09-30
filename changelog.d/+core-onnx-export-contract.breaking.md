`ONNXExportMixin.to_onnx` (behind `ImageModule.to_onnx`, `kornia.core.ImageSequential.to_onnx` and the model
exporters) now requests ONNX opset 18, the opset the dynamo exporter builds natively, and documents it. It used to
request 17: the dynamo exporter, the default of `torch.onnx.export` from torch 2.9, met that only when every operator
had a down-conversion adapter (a convolution came back at 17, a `GaussianBlur2d` pipeline at 18 after a logged
conversion failure), and the legacy exporter of older torch returned 17. Pass `opset_version=` to request another one.
To chain an export with the pre-exported Hub operators (opset 17) in `ONNXSequential`, pass
`auto_ir_version_conversion=True`; on torch before 2.9, `to_onnx(opset_version=17)` also works. On torch 2.9 and later
the IR versions already differed, so the conversion was needed before as well.
The default input shape (`ONNX_DEFAULT_INPUTSHAPE`) is now `[-1, 3, -1, -1]` instead of `[-1, -1, -1, -1]`, so a call
without `input_shape` exports a fixed 3-channel input with a dynamic batch, height and width: on torch 2.9 and later
the old default failed for any model with a fixed channel count, and on older torch it declared the channel dimension
dynamic even where the model fixed it. A channel-agnostic model exported without `input_shape` used to accept any
channel count and now takes exactly 3. A `pseudo_shape` given without `input_shape` sets the default's fixed entries,
so `to_onnx(pseudo_shape=[1, 1, 32, 32])` exports a 1-channel input. Pass `input_shape` for a dynamic channel count.
An explicit `input_shape` whose fixed entries differ from an explicit `pseudo_shape` now raises `ValueError`; the
`input_shape` entry used to win silently, so the model was traced with a shape other than the one given.
`ONNXSequential(..., auto_ir_version_conversion=True)` now converts the models to the highest opset among them instead
of to 17, so it accepts kornia's own opset-18 exports; on some of them, such as a `GaussianBlur2d` pipeline, it used to
fail with `No Adapter To Version $17 for Pad`.
