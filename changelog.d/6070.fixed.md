`SigLip2Attention` and `MoonViTAttention`, with `MoonViTEncoder` and `MoonViT`, check the trailing dimensions of an
attention mask: the last one, the keys, must be the sequence length, and in a 3-D or 4-D mask the one before it, the
queries, the sequence length or 1. A mask with a key dimension of 1, such as `(B, 1)` or `(B, N, 1)`, used to be
broadcast over every key: on the CPU and in CUDA float64 it gave the output of no mask, and in CUDA float32, float16 and
bfloat16 `scaled_dot_product_attention` raised `RuntimeError: (*bias): last dimension must be contiguous`. A query
dimension other than 1 or the sequence length raised a broadcasting `RuntimeError`. Both now raise a `ShapeError`, a
`ValueError`, that names the sequence length.
