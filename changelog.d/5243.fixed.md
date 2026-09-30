`torch.compile` can trace a function that builds a `kornia.core.TensorWrapper`: on torch 2.14 and 2.5.1 it used to fail
inside Dynamo with `RecursionError`, with or without `fullgraph=True`, because `TensorWrapper.__getattr__` looked up the
wrapper's own unset slots through itself on the instance Dynamo had not initialised yet. The wrapper's own names (its
slots, `data`, `__dict__` and the copy and pickle hooks) are no longer forwarded to the wrapped tensor, so
`copy.deepcopy` of a `TensorWrapper`, `Vector2`, `Vector3` or `Scalar` keeps the class instead of returning a plain
`Tensor`, and a deep-copied `Hyperplane` keeps its `Vector3` normal. Pickling and `torch.save` of a wrapper no longer
fail after a torch function that pickle cannot store: after `tensor ** w` they raised `PicklingError`, and after
`torch.unique(w)` `AttributeError`. The copied and pickled state now keeps only the `used_calls` entries pickle can
store, and a `copy.copy` gets its own `used_attrs` and `used_calls` sets instead of sharing them with the original.
`TensorWrapper` also gains the operators it lacked: the reflected `2 / w` and `2 // w`; `%`, `**`, `@`, `&`, `|`, `^`,
`<<` and `>>` in their forward and reflected forms (`w % 2`, `2 % w`); unary `+`, `abs(w)` and `~w`; and `float(w)`,
`complex(w)` and `operator.index(w)`. The unary operators and the conversions used to raise `TypeError`, and so did the
binary operators with a Python number or another wrapper as the other operand; with a tensor operand the binary
operators already worked through `__torch_function__`. An operator returns what the wrapped tensor's operator returns,
wrapped in the class of the left operand when that is a wrapper and otherwise in the class of the right operand.
