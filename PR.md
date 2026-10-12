## What does this pull request do?

Keeps `resize_to_be_divisible` output dimensions positive for small inputs and reads height/width from the final two axes, matching `resize`'s supported shapes. Invalid tensor ranks and nonpositive or non-integer factors now raise clear errors.

## Related issue or discussion

Fixes #5800.

## What did you check?

- `tests/geometry/transform/test_affine.py`: 82 passed, 4 skipped (CPU, float32).
- Pre-commit hooks on both changed files: passed.
- `uv build`: source distribution and wheel built successfully.

## How did AI help?

AI helped trace the shape handling and rounding edge case, add regression tests, and run the checks listed above.
