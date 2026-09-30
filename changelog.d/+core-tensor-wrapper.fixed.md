`torch.compile` can trace a function that builds a `kornia.core.TensorWrapper`: on torch 2.14 and 2.5.1 it used to fail
inside Dynamo with `RecursionError`, with or without `fullgraph=True`, because `TensorWrapper.__getattr__` looked up
the wrapper's own unset slots through itself on the instance Dynamo had not initialised yet. The wrapper's own names
(its slots, `data`, `__dict__` and the copy and pickle hooks) are no longer forwarded to the wrapped tensor, so
`copy.deepcopy` of a `TensorWrapper`, `Vector2`, `Vector3` or `Scalar` keeps the class instead of returning a plain
`Tensor`, and a deep-copied `Hyperplane` keeps its `Vector3` normal. `TensorWrapper` also gains the operators it
lacked: `2 / w`, `2 // w`, `w % x`, `w ** x`, `w @ x`, `&`, `|`, `^`, `<<`, `>>` and their reflected forms, unary `+`,
`abs(w)` and `~w`, and `float(w)`, `complex(w)` and `operator.index(w)`, which all used to raise `TypeError`. An
operator returns what the wrapped tensor's operator returns, wrapped in the class of the left operand when that is a
wrapper and otherwise in the class of the right operand. The in-place operators `+=`, `-=`, `*=`, `/=` and `//=` used to
rebind the name to a new wrapper and leave an alias holding the old values; they, and the new `%=`, `**=`, `&=`, `|=`,
`^=`, `<<=` and `>>=`, now update the wrapped tensor in place and return the same wrapper, as they do on a tensor.
