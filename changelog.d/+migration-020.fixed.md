`Boxes.pad`, `Boxes.unpad` and `Boxes.clamp` accept the unbatched `(N, 4, 2)` container the class
documents and constructs. All three assumed the batched `(B, N, 4, 2)` indexing: `pad` and `unpad`
broadcast the padding as `(B, 1, 1)`, which does not fit the unbatched `(N, 4)` coordinate view and
raised `RuntimeError: output with shape [1, 4] doesn't match the broadcast shape [1, 1, 4]`, and `clamp`
materialized its bounds with `repeat(1, _data.size(1), 4)` and raised `IndexError: too many indices for
tensor of dimension 2`. Adding a leading singleton batch axis to the same boxes made all three work.
They now broadcast against whichever rank the container holds and match the singleton-batch result;
an unbatched container carries one image, so a per-image tensor with more than one row is rejected.
Batched results are byte-identical, including for a non-finite bound: `clamp` stays comparison-based, so
a `NaN` bound leaves the coordinate alone rather than propagating into it (closes #4244). (#4372)
