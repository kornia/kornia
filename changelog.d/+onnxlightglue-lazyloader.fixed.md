`OnnxLightGlue` now goes through the `onnxruntime` lazy loader instead of probing the module with
`importlib.util.find_spec`, so `kornia_config.lazyloader.installation_mode` applies to it. The probe returned before
anything reached `kornia.core.external.onnxruntime`, so with the mode set to `"auto"` the constructor raised
`ImportError` and installed nothing, and with `"ask"` it never asked, while every other `onnxruntime` consumer
installed `kornia[onnx]` or prompted. Under the default `"raise"` mode nothing changes: a missing `onnxruntime` still
raises an `ImportError` naming the extra to install.