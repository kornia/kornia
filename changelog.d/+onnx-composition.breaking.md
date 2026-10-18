`ONNXSequential` given `target_opset_version` or `target_ir_version` without `auto_ir_version_conversion=True` now
converts the models; the targets used to be ignored. Where the ONNX version converter has no adapter for an operator
(for example `Resize` or `Pad` from opset 18 to 17), the call now raises the converter's error, where it used to return
the unconverted graph, which ran.
