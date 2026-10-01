`solve_quartic` now solves a row as a cubic only when its leading coefficient is below the tolerance and the
scale-invariant root bound `max(|b/a|, |c/a|^(1/2), |d/a|^(1/3), |e/a|^(1/4))` exceeds `1 / tol`, so a quartic
whose roots are all below `1 / (4 * tol)` in magnitude keeps them at every scale. Rows such as
`[1e-7, 0, 0, 0, -1]` previously returned `[0, 0, 0, 0]` and now return their real roots `±56.23`;
`(x-50)(x-60)(x-70)(x-80)` scaled by `2^-21` in float32 now returns all four roots instead of a single root
near 31.5. (#4954)
