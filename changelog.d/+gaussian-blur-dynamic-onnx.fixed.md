`GaussianBlur2d` exported with the legacy ONNX tracer now accepts a dynamic batch, height and width. The export used to fail because the convolution kernel had no static shape.
