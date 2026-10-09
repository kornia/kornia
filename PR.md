## Summary

Fix Dice loss for empty weighted reductions. Fully ignored samples and reductions with zero weighted cardinality now produce loss 1 with finite gradients, including when `eps=0`. This also covers float16 weights that zero out the cardinality with the default epsilon.

Closes #5631

## Verification

- `pixi run -e default test-module tests/losses/test_dice.py` — 53 passed
- `pixi run -e default pre-commit-all` — passed
- `pixi run -e default uv build --no-sources` — source distribution and wheel built
