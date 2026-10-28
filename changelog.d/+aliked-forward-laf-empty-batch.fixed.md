`ALIKED.forward_laf` now returns empty `(0, 0, 2, 3)` LAFs, `(0, 0, 1)` responses and `(0, 0, D)` descriptors for an
empty batch instead of raising "stack expects a non-empty TensorList" (#6266).
