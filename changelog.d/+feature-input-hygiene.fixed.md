`match_adalam(dm=...)` now uses the distance matrix for the nearest-neighbour matches and their ratio scores; the argument was documented but never read, so any `dm` gave the same matches as no `dm`. Without `dm` the squared L2 distances are used, as before.

`match_adalam` with a `config` that does not set `"device"`, and `GeometryAwareDescriptorMatcher("adalam")`, now run on the device of the descriptors; they used to move everything to the CPU and return CPU tensors for CUDA or MPS input. The unique-pair step no longer needs `torch.unique(dim=0)`, which torch 2.5 lacks on MPS.

`match_adalam(hw1=..., hw2=...)` now honours the image sizes when they are lists; only tuples were used and a list silently fell back to the keypoint spread.
