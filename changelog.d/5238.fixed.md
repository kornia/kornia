`get_gaussian_discrete_kernel1d` uses exponentially scaled Bessel terms to avoid NaN kernels from
overflow at larger sigma values. Half-precision inputs use float32 intermediates and retain their
original output dtype.
