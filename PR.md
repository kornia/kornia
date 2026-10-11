## What does this pull request do?

Fixes #5769. Replaces the depth `Tensor.unfold` in the 3D convolutional soft-argmax window construction with equivalent strided depth slices. The legacy TorchScript ONNX exporter cannot lower that `unfold` because it cannot access the traced input size.

Adds static and dynamic ONNX Runtime regression coverage for both the functional API and `ConvSoftArgmax3d`.

## What did you check?

- `tests/geometry/subpix/test_spatial_softargmax.py` on CPU, float32 and float64, with PyTorch 2.5.1: 378 passed, 20 skipped.
- The same focused test module on CPU, float32 and float64, with PyTorch 2.14.0: 378 passed, 20 skipped.
- New legacy ONNX export tests (function/module, static/dynamic shapes) on PyTorch 2.5.1 and 2.14.0: 4 passed on each version.
- Wheel build completed successfully.
- Pre-commit checks passed after its mixed-line-ending hook normalized the two changed Python files.
