Reduce local-feature extraction overhead by accumulating orientation histograms directly,
batching built-in DoG extrema refinement, selecting coordinates before combining response signs,
and using faster activation layouts in KeyNet and CPU HardNet. Detector settings and pretrained
checkpoint formats are preserved. On CUDA the orientation histogram now accumulates with atomics,
so identical inputs can differ at the ulp level between calls unless
`torch.use_deterministic_algorithms(True)` is set. (#4254)
