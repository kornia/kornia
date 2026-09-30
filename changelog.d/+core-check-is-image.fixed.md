`KORNIA_CHECK_IS_IMAGE` checks the shape again with the default `raises=True`, as its docstring says and as it did
before #3156: a tensor that is not `(*, 1, H, W)` or `(*, 3, H, W)`, such as `(5, 4, 5)`, `(4, 5)` or a 0-d scalar,
now raises `ImageError` instead of returning `True`. Code that passed a non-image through the check therefore gets
`ImageError` now, and so does `image_to_string`, which rendered a 6-channel tensor as if it had 3 channels.
`Image.print` now moves channels-last data to channels-first before rendering it, as `Image.write` does, so an image
from `Image.from_numpy` prints correctly instead of as a garbled picture. The value
check is fixed as well. A NaN value fails the range check. An empty image, such as a `(0, 3, H, W)` batch, passes
instead of crashing inside `torch.aminmax`, with `raises=False` too. A signed integer image whose `bits` is the dtype's
width (`int8` with the default `bits=8`, `int64` with `bits=64`) is no longer rejected for every value, since
`2 ** bits - 1` is no longer converted to the dtype, where it overflowed. `uint16`, `uint32` and `uint64` images are
range-checked instead of crashing. The integer range error states `[0, 2 ** bits - 1]` and sets `expected_range` to it,
where it used to say `[0, 1]` whatever `bits` was.
