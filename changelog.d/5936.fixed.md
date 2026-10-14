`So3.matrix()` (and `adjoint()`), `So3 * So3`, `So3 * points`, `Se3 * Se3` and `Se3 * points` now compute in float32
for float16 and bfloat16 inputs and return the input dtype, rounding once; `Se3.matrix()` and `Se3.inverse()` follow
through `So3`. Evaluated in half precision, each drifted 3 to 6 units of roundoff from the exact result, and
identities such as `R.T @ R == I`, `(s * s.inverse()).t == 0` and chained point transforms missed by up to about 12
units; they now hold to about 2. float32 and float64 results are unchanged. Building a bfloat16 `So2` or `Se2`
(`exp`, `identity`, `random`, `from_matrix`, `Se2.trans`, `trans_x`, `trans_y`, and a `NamedPose` of an `Se2` pose)
used to fail inside `torch.complex` with a `RuntimeError`, since PyTorch has no complex bfloat16 dtype; it now raises
`TypeCheckError` naming the limitation. `.bfloat16()` on a built `So2` or `Se2` keeps working and keeps the
rotation's complex precision.
