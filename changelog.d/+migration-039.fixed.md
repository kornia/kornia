`solve_quartic` now validates Ferrari resolvent candidates with a dtype-aware scaled residual,
falls back to the smallest scaled residual when no candidate passes, and keeps the Ferrari `E`
factorization away from an ill-conditioned `R` division. Cancellation-level `R^2` is snapped to
zero relative to the coefficient scale, and all Ferrari compute dtypes choose between the linear-
and constant-term `E` forms by normalized coefficient error; float32 also refines the selected
resolvent root once. The gradient-safe zero-radicand path remains intact, preventing half-precision
non-finite or dropped roots and full-precision non-roots while preserving the public contract. (#4357)
