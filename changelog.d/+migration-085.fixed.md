`HyNet` and `SOSNet` now run in half precision, on CPU and on GPU, and no longer return NaN for a
degenerate patch (closes #4224). Two defects sat on the same line. On CPU both raised
`NotImplementedError: "avg_pool3d_out_frame" not implemented for 'Half'` (and the `'BFloat16'` spelling):
their final `LocalResponseNorm` is handed a 4-D `(B, C, 1, 1)` tensor, which routes
`torch.nn.functional.local_response_norm` through `avg_pool3d`, and that kernel has no CPU
`float16`/`bfloat16` implementation. Separately, in `float16` on a device where the kernel does exist
(MPS, CUDA), the descriptors came back all-NaN for any patch the network maps to exactly zero: the `eps`
that keeps the normalisation's division defined (`SOSNet.forward`'s `eps`, `HyNet`'s `eps_l2_norm`, both
`1e-10`) is not representable in `float16` and flushes to `0.0`, leaving `0/0`. Every `Conv2d` in `SOSNet`
has `bias=False` and every `BatchNorm2d` is `affine=False`, so any constant patch reaches the
normalisation as exactly zero; `HyNet` reaches it only with `is_bias=False`. `bfloat16` keeps `float32`'s
exponent range, so the guard survives there and only the CPU kernel gap applied to it. Both models now
take that one normalisation step in `float32` for either half-precision input and cast the result back --
a wider lift than the `float16`-only one `kornia.feature.siftdesc` gives its own `1e-10` guards, because
the CPU kernel gap covers both dtypes. `float32` and `float64` take the original expression and are
bitwise unchanged on CPU, CUDA and MPS. Half-precision output does change, by 0.25-2.25 eps, and where it
moves most it moves towards the `float64` reference rather than merely away from NaN: `SOSNet` `float16`
goes from 2.15e-03 to 2.28e-04 of maximum absolute error against a `float64` model carrying the same
weights, and the configurations that do not improve stay within one eps of where they were.
Half-precision descriptors are therefore not comparable bit-for-bit across this release. (#4225)
