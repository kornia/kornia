`KORNIA_CHECK_IS_IMAGE` range-check fixes. An empty image, such as a `(0, 3, H, W)` batch, passes instead of crashing
inside `torch.aminmax`, with `raises=False` too. A signed integer image whose `bits` is at least the dtype's width (for
example `int8` with the default `bits=8` or with `bits=9`, or `int64` with `bits=64`) used to fail for every value,
because `2 ** bits - 1` was converted to the dtype, where it overflowed. It is now range-checked normally. `uint16`,
`uint32` and `uint64` images are range-checked instead of crashing. The integer range error states
`[0, 2 ** bits - 1]` and sets `expected_range` to it, where it used to say `[0, 1]` whatever `bits` was.
`Image.print` now moves channels-last data to channels-first before rendering it, as `Image.write` does, so an RGB image
from `Image.from_numpy` prints correctly instead of as a garbled picture.
