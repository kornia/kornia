## Summary

Validate the positional input count in the ONNX Runtime mixin before mapping values to session input names. Calls with too few or too many arrays now raise a clear `ValueError`, preventing silent use of only a prefix of the supplied inputs.

Closes #5798

## Verification

- `pixi run -e default test-module tests/core/test_onnx_mixin.py tests/onnx/test_sequential.py` — 24 passed
- `pixi run -e default pre-commit-all` — passed
- `pixi run -e default uv build --no-sources` — source distribution and wheel built
