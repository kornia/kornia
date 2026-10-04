`So2` module dtype conversions now preserve both complex rotation components:
`float()` and `to(float32)` use complex64, while `double()` and `to(float64)` use
complex128. Parent modules and `Se2` inherit the same behavior; device-only moves
preserve precision. Complex checkpoint keys and shapes remain unchanged.
Half precision uses PyTorch's experimental complex32 support; bfloat16 conversion
raises because PyTorch has no corresponding complex dtype.
