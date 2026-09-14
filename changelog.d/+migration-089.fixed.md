`quaternion_exp_to_log` and `euler_from_quaternion` no longer return `nan`/`inf` gradients at their
respective rotation singularities (#4007). Both differentiate an inverse trig function (`acos`, `asin`)
whose own derivative is unbounded at its domain boundary (`w = +-1`, `sinp = +-1`); `quaternion_exp_to_log`
hits that boundary exactly at the identity quaternion `(1, 0, 0, 0)` -- the standard initialization for
pose optimization -- where the unbounded derivative used to multiply the identity's exactly-zero vector
part into `0 * inf = nan`, and `euler_from_quaternion`'s `pitch` hits it at gimbal lock. Both now guard the
argument fed to the differentiated `acos`/`asin` call away from the boundary while taking the returned
*value* from a detached copy of the real (possibly boundary) argument, so the forward result is unchanged
-- verified byte-identical against the previous implementation over 500 random inputs including several
forced onto the exact boundary -- while the gradient no longer depends on what a given torch version or
backend's own `acos`/`asin` derivative happens to return there. `quaternion_exp_to_log`'s antipode
`(-1, 0, 0, 0)` still returns a large but finite gradient (`~pi/eps`) from an unrelated, pre-existing
division by its clamped (zero) norm; that is unchanged and out of scope here.
