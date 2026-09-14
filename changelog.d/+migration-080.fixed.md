`quaternion_to_axis_angle` returns the correct gradient at every identity-like quaternion, not only the
positive unit identity `(1, 0, 0, 0)` (#4237). Its zero-vector-part branch used the constant `k = 2.0`,
which is only the `w = 1` case of the true analytic limit `k = 2 / w` (`w` = the real component); every
other identity-like input got a wrong gradient. That included the negative unit identity `(-1, 0, 0, 0)`
-- the same physical rotation as `(1, 0, 0, 0)` under the double cover -- which got the **wrong sign**
(`+2` instead of `-2`), and any non-unit "identity" such as `(2, 0, 0, 0)` (a scale this function
explicitly permits), which got the **wrong magnitude** (`+2` instead of `+1`). This was a genuine
backward-correctness defect, not a missing edge case: a valid unit quaternion could produce the opposite
gradient direction in pose optimization depending purely on which sign of the identity it started from.
The prior regression test for the `#3949` NaN-gradient fix only exercised `w = 1`, the one point where
the wrong constant and the correct formula happen to agree, and so could not have caught this. The
division needed for `2 / w` is guarded at `w = 0` the same way the function's existing `sin_theta`
division already is. That gate is `~pos` -- the mask that actually selects the branch -- and not merely
`w != 0`: `2.0 / t` lowers to `t.reciprocal() * 2`, whose backward is `-grad * result**2`, and for a
rotation just short of a half turn `w` is small but non-zero, so `(1 / w)**2` overflows to `inf` and
meets the exact `0.0` that the unselected branch receives as `0 * inf` -> `nan`. In `float16` that
would have been every rotation within ~0.45 degrees of 180 -- ordinary inputs, not degenerate ones.
The `2 / w` coefficient is also detached, because on the branch that selects it the vector part is
zero, so `d(out)/dw` is exactly zero there and computing it through `-2 / w**2` only reintroduces the
same overflow.

The fully degenerate all-zero quaternion `(0, 0, 0, 0)` -- not a valid rotation -- keeps its
`(0, 0, 0)` forward value. Its gradient there changes, and improves: it was `(0, 2, 2, 2)` on torch
2.14 but already `(nan, 2, 2, 2)` on torch <= 2.9.1, where `atan2(0, 0)`'s derivative with respect to
its second argument -- a `0 / 0` -- returns `nan`. `atan2` is now shielded across that whole masked
branch, which also clears a pre-existing `nan` for a zero vector part whose `w**2` underflows
(`w = 1e-30` in `float32`, `|w| < 2.4e-4` in `float16`). No gradient is analytically correct at a point
with no well-defined limit, so the changed value is not a regression.

The forward value is unaffected everywhere: the `atan2` shield deliberately takes its value from the
unshielded expression, because routing `cos_theta` through `torch.where` hands `atan2` a contiguous
tensor instead of a stride-4 view and can select a different kernel. Verified byte-identical against
the previous implementation over 2000 random inputs in `float64`, `float32`, `float16` and `bfloat16`,
including forced unit, negative-unit, scaled, all-zero and near-half-turn quaternions, with no
non-finite gradient row remaining at any of those dtypes.
