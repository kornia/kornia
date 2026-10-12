`EfficientViT` now runs with float16 weights on CPU. `LiteMLA.relu_linear_att` lifts a float16 input to float32 for its
matrix products and returned the float32 result to a float16 projection layer, which raised
`RuntimeError: expected scalar type Float but found Half`. The result is now converted back to the input dtype.
float32 and bfloat16 results are unchanged.
