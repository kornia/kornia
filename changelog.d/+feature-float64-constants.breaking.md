`CornerHarris` with a number `k` now holds it as a Python `float`, not as a 0-dimensional float32 buffer. So
`harris.k.item()` raises `AttributeError`; a tensor assigned to `harris.k` stays a plain attribute, which `.to()` does
not move and the `state_dict` does not hold; a subclass that registers its own `k` buffer after passing a number to
`CornerHarris.__init__` raises `KeyError`; and the `state_dict` no longer holds `k`, so a strict load into an older
kornia raises a missing-key `RuntimeError`. A tensor `k` is still stored as a buffer, as before.
