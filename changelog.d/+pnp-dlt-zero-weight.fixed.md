`solve_pnp_dlt` now applies `weights` to the point normalization and the degeneracy check as well as
to the linear system, so a zero weight is the same as removing the point. Coplanar points plus
zero-weight points off the plane now raise instead of returning a wrong pose. Uniform weights give
the same result as before.
