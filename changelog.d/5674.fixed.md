`extract_tensor_patches` and `combine_tensor_patches`, including their module wrappers,
now handle empty image batches without ambiguous reshape errors. Patch counts, spatial
sizes, dtype, device and autograd connections are preserved. `PatchSequential` also
supports empty batches when its children do, preserving the existing `same` padding
and `valid` cropping behavior. Geometric patch inverses and mask transforms remain
unsupported. Part of #4429.
