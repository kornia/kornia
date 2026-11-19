`combine_tensor_patches` reuses one spatial overlap-count map across batches and channels, reducing reconstruction time and temporary tensor allocations without changing the overlap average.
