Lovasz hinge loss sorts target labels without allocating a repeated batch-index tensor, saving 8 bytes per input pixel while preserving values and gradients.
