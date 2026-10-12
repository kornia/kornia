The ONNX guide and the landing page now export with the `torch.export`-based exporter (`dynamo=True`, with
`dynamic_shapes`) instead of the deprecated TorchScript-based one at opset 17, and show the
`auto_ir_version_conversion=True` flag that `ONNXSequential` needs to chain such an export with the pre-exported Hub
operators.
