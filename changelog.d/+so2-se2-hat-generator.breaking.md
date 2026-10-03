`So2.hat` / `So2.vee` and `Se2.hat` / `Se2.vee` now use the standard Lie algebra generators. `So2.hat(theta)` used to
return the symmetric `[[0, theta], [theta, 0]]` and `So2.vee` read its `[0, 1]` entry; it now returns
`[[0, -theta], [theta, 0]]` and `vee` reads `[1, 0]`. `Se2.hat(v)` used to put `(v_x, v_y)` in the bottom row
below that symmetric block; it now returns `[[0, -theta, v_x], [theta, 0, v_y], [0, 0, 0]]`, and `Se2.vee` reads the
translation from the last column. `vee(hat(v))` is still the identity, and `matrix_exp(hat(v))` now equals
`exp(v).matrix()`, as it already did for `So3` / `Se3`. Code that builds or reads these matrices in the old layout
gets different values. (#4929)
