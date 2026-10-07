# fix(augmentation): track shapes through nested containers

## Summary

- Preserve the child module while recursively tracking shapes through nested augmentation containers, so export-time size changes use the nested module's static size.
- Track `PatchSequential(padding="valid")` cropping from its grid and padding configuration, including empty patch pipelines.
- Add regression coverage for nested `torch.export` resize/crop pipelines and nested valid-padding patch sequences.

Fixes #5596.

## Validation

- `pixi run pre-commit-all`
- `pixi run test-module tests/augmentation/container/test_patch_sequential.py` (430 passed, 182 skipped)
- `pixi run test-module tests/augmentation/test_torch_export.py` (13 passed)
- `pixi run uv build` (source distribution and wheel built)

The PR description is committed here as requested.
