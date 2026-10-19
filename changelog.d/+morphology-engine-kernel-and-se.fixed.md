`kornia.morphology.erosion` and the composites built on it (`opening`, `closing`, `gradient`, `top_hat`,
`bottom_hat`) give the same result for a bool or integer `structuring_element` as for the equal float one, on
every engine. The `shift` engine, which `engine="auto"` picks off CUDA, used to negate the structuring element
in its own dtype, which wraps: a `uint8` entry of 2 became an offset of +254 instead of -2, and an `int8` entry
of -128 an offset of -128 instead of +128. A bool structuring element raised `NotImplementedError` under
`unfold`, `shift` and `auto`. `dilation`, `erosion` and the composites also accept a non-contiguous `kernel`,
such as a transposed view, under `engine="convolution"`, which used to raise `RuntimeError: view size is not
compatible with input tensor's size and stride`. Results for float kernels and structuring elements are
byte-identical to before.
