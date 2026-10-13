`kornia.core.utils.dict_to_dataclass` now converts nested dicts in fields annotated `Optional[...]`, other unions,
`list`, `tuple` and `dict` of a dataclass, and in string annotations (including modules with
`from __future__ import annotations`), and keeps a value that is already an instance of the field's dataclass
instead of raising `TypeCheckError`. Integer and bool tensors passed to `ParametrizedLine`, `Se2`, `Se3` and the
other constructors built on `register_module_state` are stored as buffers with their dtype instead of failing
with torch's "Only Tensors of floating point and complex dtype can require gradients" error. The API reference
now renders the `kornia.core.utils` device and autocast helpers that the `kornia.utils` deprecation warnings
point to.
