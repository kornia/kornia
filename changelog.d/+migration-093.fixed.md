Fix `Boxes.merge` concatenating along the vertex axis instead of the box axis
for unbatched `(N, 4, 2)` input. The old `dim=1` was correct for batched
`(B, N, 4, 2)` data (where dim 1 is the box axis) but wrong for 3-D tensors
(where dim 1 is the vertex dimension); `dim=-3` is correct in both cases and
the behaviour of the batched path is byte-identical. For list-backed inputs,
each batch row is now repacked with its real boxes before all trailing padding,
and the combined per-image padding counts are retained. This makes ``to_tensor``
trim only padding instead of dropping merged boxes or exposing old padding as
real boxes (#4168, #4175).
