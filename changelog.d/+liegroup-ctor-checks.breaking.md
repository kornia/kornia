`So2`, `Se2` and `Se3` now reject at construction the inputs their methods cannot use. `So2(z)` with a real
floating-point `z` used to construct, and `inverse()`, `adjoint()`, `__getitem__` and `So2 * So2` worked on it while
`matrix()`, `log()` and `So2 * points` raised torch's "imag is not implemented"; it now raises `TypeCheckError`
naming `z` and its dtype (build `z` with `torch.complex`). `Se2` / `Se3` with a translation whose dtype differs from
the rotation's (for example a float64 rotation and a float32 translation) used to construct, and `matrix()`,
`inverse()`, `__mul__` and `__getitem__` worked on some of them while `log()` (`Se2`) or `inverse()`, `adjoint()` and
`Se3 * Se3` (`Se3`) raised torch's "expected scalar type" error; they now raise `TypeCheckError`. A translation on
another device than the rotation, which failed in every method, now raises `DeviceError`. A bfloat16 `Se2`
translation beside a complex rotation stays accepted, because `.bfloat16()` leaves exactly that state.
