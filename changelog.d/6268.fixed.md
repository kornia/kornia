`ColorJiggle`'s channel order and `RandomMixUpV2`'s pairs are drawn as an int64 permutation, whatever the dtype the
parameter generator samples in. A generator set to `bfloat16` used to raise `"randperm_out_cuda" not implemented for
'BFloat16'` on CUDA. That happened through `set_rng_device_and_dtype`, through `.to(torch.bfloat16)` on the
augmentation, and, since containers and parent modules now pass `.to()` on to their augmentations, through
`.to(torch.bfloat16)` on any model that contains one. In `float16`, a MixUp batch above 2048 could not be permuted
exactly. torch draws the same permutation for every dtype, so the parameters of every call that worked before are
unchanged.
