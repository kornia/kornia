`equalize_clahe` and `RandomClahe` accept a `grid_size` whose two entries differ. Such a grid used to raise
`IndexError: shape mismatch: indexing tensors could not be broadcast together` on every image (#2531). Output
for square grids is unchanged.
