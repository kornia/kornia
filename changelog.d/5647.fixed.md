Empty-batch pipelines containing `CenterCrop` now generate subsequent-stage
parameters without indexing empty probability or output-size tensors. Shape
tracking follows the current image-forward probability behavior, including the
existing empty-gate fallback tracked in #4429.
