`torch.compile` can trace a function that builds a `kornia.core.TensorWrapper`: on torch 2.14 and 2.5.1 it used to fail
inside Dynamo with `RecursionError`, with or without `fullgraph=True`, because `TensorWrapper.__getattr__` looked up
the wrapper's own unset slots through itself on the instance Dynamo had not initialised yet. The wrapper's own names
(its slots, `data`, `__dict__` and the copy and pickle hooks) are no longer forwarded to the wrapped tensor, so
`copy.deepcopy` of a `TensorWrapper`, `Vector2`, `Vector3` or `Scalar` keeps the class instead of returning a plain
`Tensor`, and a deep-copied `Hyperplane` keeps its `Vector3` normal. Pickling, `torch.save` and `copy.deepcopy` of a
wrapper no longer fail with `PicklingError` after a torch function that pickle cannot store, such as `torch.unique(w)`
or `tensor ** w`: the copied and pickled `used_calls` now keep only the functions pickle can store. `TensorWrapper` also
gains the operators it lacked: `/`, `//`, `%`, `**`, `@`, `&`, `|`, `^`, `<<` and `>>` in their forward and reflected
forms (`w % 2`, `2 / w`), unary `+`, `abs(w)` and `~w`, and `float(w)`, `complex(w)` and `operator.index(w)`. The unary
operators and the conversions used to raise `TypeError`, and so did the binary operators with a Python number or
another wrapper as the other operand; with a tensor operand the binary operators already worked through
`__torch_function__`. An operator returns what the wrapped tensor's operator returns, wrapped in the class of the left
operand when that is a wrapper and otherwise in the class of the right operand.
