`kornia.image.Image` now raises `ShapeError` when its `ImageLayout` gives a height, width or channel count that the data does not have, whatever the type of the size. In 0.9.0rc1 the layout check compared the sizes as strings and skipped every size whose `str()` is not all digits, a rule meant for dimension names. So a tensor, float or negative size was never compared:

- `ImageSize(torch.tensor(9), torch.tensor(9))`, `ImageSize(9.0, 9.0)`, `ImageSize(4.5, 5.0)` and `ImageSize(-1, -1)` were all accepted for a 4 x 5 image.
- With the 9 x 9 sizes, `height` and `width` then reported 9, not the data's 4 and 5.

Python int sizes that match the data are accepted as before.

The message has changed. A size mismatch used to read `Shape mismatch at dimension 1: expected 9, got 4.`, followed by `Expected shape:` and `Actual shape:` lines. A wrong number of dimensions read `Shape dimension mismatch: expected 3 dimensions, got 4.` Both now read like `Shape mismatch: expected (3, 9, 9), got [3, 4, 5].`, and a tensor size prints as `tensor(9)`.

Building an `Image` from `x.shape` under `torch.compile(fullgraph=True)` now also compiles at a second image size. At that size torch recompiles with dynamic shapes, and turning the symbolic size into a string raised `Unsupported: tp_repr_impl not implemented`.
