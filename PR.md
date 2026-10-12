Fixes #5768

## What does this pull request do?

Apply independently optional coarse masks in LoFTR's dual-softmax and Sinkhorn matching paths. A missing mask now leaves that image's cells valid, while masked cells remain excluded from the confidence matrix and returned matches.

## What did you check?

- Focused one-sided mask and existing masked-pair tests: 12 passed on CPU with Torch 2.14 and 2.5.1.
- Full `tests/feature/test_loftr.py`: 37 passed, 8 skipped on CPU float32.
- Changed-file pre-commit hooks passed.
- Built the Kornia wheel successfully with `uv build --wheel`.
