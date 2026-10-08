For an empty batch after an always-applied `RandomResizedCrop`, later stages now
record the crop's output spatial size rather than its input size. Intermediate
inverses consequently restore the size received by each stage. Downstream
`CenterCrop` size validation now checks that actual spatial size. Regenerate saved
parameter lists for affected empty pipelines; previously saved metadata is not
rewritten. Non-empty parameter generation is unchanged. Part of #4429.
