`kornia.losses.total_variation` reduces over one flattened dimension instead of `dim=(-2, -1)`, which returns identical values and avoids a slow reduction path on MPS.
