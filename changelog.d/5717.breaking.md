`Hyperplane` now saves and restores its normal and offset in `state_dict()` and moves them with `.to()`.
Previously these tensors were omitted from module state. Loading an older empty plane checkpoint with
`strict=True` now reports the missing `_n` and `_d` keys; recreate the plane from its original geometry.
