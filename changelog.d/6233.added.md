The ONNX, torch.compile and torch.export page now opens with "What Kornia promises": the stable-core configurations
that export with the dynamo ONNX exporter at opset 18 and match eager in onnxruntime, and the ones that compile with
`torch.compile(..., fullgraph=True)` and match eager, for inference at static shapes, checked in CI on torch 2.9 and
2.14, with the configurations left out, the known failures and their tracking issues; the survey tables mark every
promised configuration. `nms` and `draw_rectangle` document how `torch.compile` handles them, and are left out of the
compile promise (#6194).
