`SemanticSegmentation.visualize` works for CUDA, MPS and half-precision models. It indexed the
CPU-drawn colormap with the mask's `argmax`, which raises for CUDA and MPS masks, and recognised a
softmax head with `torch.allclose(sum, 1)` at float32-sized default tolerances, which a float16 or
bfloat16 sum of probabilities never meets; the tolerance now scales with the number of classes and the
dtype's epsilon, and the colormap follows the mask's device and dtype, so the visualization keeps the
model's dtype instead of coming back as float32. `OnnxLightGlue` without `onnxruntime` raises an
`ImportError` that names `pip install "kornia[onnx]"`, like the lazy-loader handles do, instead of a
bare `BaseError`. (#4301)
