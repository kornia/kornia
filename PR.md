## What does this pull request do?

Fixes #5774. Promotes half-precision inputs to float32 before subtraction in `welsch_loss`, then casts the result back to the promoted input dtype. This prevents finite float16 inputs with an overflowing residual from producing NaN gradients.

Adds a regression test for the reported large residual and updates the existing half-precision gradient reference to calculate from the more accurate float32 intermediate.

## What did you check?

- `tests/losses/test_welcsh.py` and `tests/losses/test_robust.py` on CPU with float16, bfloat16, float32, and float64 using PyTorch 2.5.1: 165 passed, 15 skipped.
- The same tests on CPU using PyTorch 2.14.0: 165 passed, 15 skipped.
- Pre-commit checks passed.
- Wheel build completed successfully.
