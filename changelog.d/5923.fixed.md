`RandomMotionBlur` and `RandomMotionBlur3D` with a ranged `kernel_size` accept an empty batch and return it unchanged.
They raised before the blur ran because they choose the batch's kernel size from the sampled sizes, of which an empty
batch has none: `RandomMotionBlur` with `IndexError: list index out of range` and `RandomMotionBlur3D` with
`RuntimeError: a Tensor with 0 elements cannot be converted to Scalar`.
