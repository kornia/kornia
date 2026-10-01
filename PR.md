# Fix boundary darkening in edge-aware blur pooling

## Summary

- Keep the edge detector's 2-pixel reflect halo independent from the blur halo.
- Reflect-pad by the blur kernel radius before `blur_pool2d`, preventing its zero padding from affecting constant image boundaries.
- Add regression coverage for kernel sizes 3, 5, 7, and 9.

Fixes #5228.

## Validation

- Focused tests: `pytest tests/filters/test_blur_pool.py -q` (78 passed).
- `pre-commit run --all-files` (all hooks passed).
- `uv build` (wheel and source distribution built successfully).
