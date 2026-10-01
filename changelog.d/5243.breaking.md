The in-place operators of `kornia.core.TensorWrapper`, and so of `Vector2`, `Vector3` and `Scalar` (`+=`, `-=`, `*=`,
`/=`, `//=`, `%=`, `**=`, `&=`, `|=`, `^=`, `<<=` and `>>=`), now update the wrapped tensor in place and return the same
wrapper, as they do on a tensor. They used to rebind the name to a new wrapper: `+=`, `-=`, `*=`, `/=` and `//=` with
any operand, and the others with a tensor operand (with a Python number or another wrapper they raised `TypeError`).
So an alias kept the old values, and the tensor the wrapper was built from was never modified. A wrapper does not copy
the tensor it is built from, so `v = Vector3(t); v += 1` now also modifies `t`. An in-place operator now raises
`RuntimeError` where the tensor's in-place operator raises, where it used to return a new wrapper: on a wrapper of a
leaf that requires grad, such as `Vector3(nn.Parameter(...))` ("a leaf Variable that requires grad is being used in an
in-place operation"), for an update that changes the dtype, such as `/=` on an integer wrapper ("result type Float
can't be cast to the desired output type Long"), and for an update that broadcasts to a larger shape. Write
`v = v + x` to keep the old behaviour. `@=` still rebinds, as it does on a tensor.
