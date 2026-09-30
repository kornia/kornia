`to_onnx` exports the model in eval mode and restores each submodule's training flag afterwards, also when the export
fails. On torch 2.9 and later a model left in training mode exported an active `Dropout`, so onnxruntime zeroed about
half of the outputs; the legacy exporter of older torch exported in eval mode but then reset every submodule to the
model's own mode. The dummy input takes the dtype and device of the model's first floating parameter or, for a model
without one, of its first floating buffer, so float64, float16 and MPS or CUDA models export with an input of their
own dtype. The dummy input used to be float32 on the CPU, which failed on an input/weight mismatch or, for a model that
promotes, exported a float32 input. A dynamic `input_shape` whose rank differs from the pseudo shape now raises
`ValueError`: a longer one raised `IndexError`, and a shorter one exported, sized silently from the pseudo shape's
leading entries. A missing output directory raises `FileNotFoundError` before the export runs rather than after it,
and the export no longer passes through the deprecated `io.BytesIO` destination or, on the dynamo exporter,
`dynamic_axes`, which warned on every call. `ONNXSequential` and `ONNXModule` now use the `session_options=` they are
given instead of raising `UnboundLocalError`, and `kornia.onnx.utils.add_metadata` (behind `to_onnx` and the
`add_metadata` methods) overwrites a key that is already present instead of adding a duplicate that
`onnx.checker.check_model` rejects.
