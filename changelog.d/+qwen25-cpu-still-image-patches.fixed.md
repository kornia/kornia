`Qwen25VisionTransformer` projects repeated still-image patches on the CPU without materializing the temporal image copies, retaining projection hooks, checkpoint names, and gradients.
