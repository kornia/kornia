`KORNIA_CHECK_IS_IMAGE` checks the shape again with the default `raises=True`, as its docstring says and as it did
before #3156. It used to return `True` for a tensor that is not `(*, 1, H, W)` or `(*, 3, H, W)`, such as `(5, 4, 5)`,
an RGBA `(B, 4, H, W)`, `(4, 5)` or a 0-d scalar, while the same call with `raises=False` returned `False`. It now raises
`ImageError`, so code that passed a non-image through the check gets `ImageError`. So does `image_to_string`, which
rendered a 6-channel tensor as if it had 3 channels. A float image containing NaN used to pass the range check for
both `raises` values. It now fails it: `raises=True` raises `ValueCheckError` and `raises=False` returns `False`.
