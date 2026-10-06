## What does this pull request do?

Fixes the image-module output cache edge cases in #5211. `show()` and `save()` now report a clear error before an image
exists or when the output is not a 3D/4D image tensor. They preserve singleton image dimensions, support bfloat16 output,
and use the existing PIL conversion path. Disabling image features clears the cached output, and pickling omits it.

## Related issue or discussion

Fixes #5211

## What did you check?

- `tests/core/test_module.py` on CPU with float16, bfloat16, float32, and float64: 275 passed, 14 skipped.
- Related augmentation-container cache and rendering tests: 10 passed.
- `uv build --wheel`: succeeded (`kornia-0.9.0rc1-py3-none-any.whl`).
- `git diff --check`: passed.

## How did AI help?

AI helped trace the shared image conversion and cache paths, reproduce the reported edge cases, and draft regression tests.
The focused test suites and wheel build were run locally against the changed worktree.
