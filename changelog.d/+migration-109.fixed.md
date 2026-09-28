Fix 2-D, 3-D, and mix augmentations raising a cross-device `RuntimeError` when their random parameters are generated
on the default CPU device and their input is on MPS or CUDA (#4164). Since #3722, four branchless `torch.where`
blends used the CPU `batch_prob` mask directly with accelerator outputs or transformation matrices. The selection
mask now moves to the output device at each blend boundary, preserving CPU random-number generation and the
ONNX/fullgraph-friendly blend. Calling `augmentation(x_accelerator)` therefore works again without first moving the
augmentation module or its RNG to the accelerator (#4151).
