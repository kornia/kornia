The `RANSAC(compile=True)` documentation now limits the CPU-then-CUDA inductor-cache segfault to PyTorch before 2.15. PyTorch 2.15.0 saves a separate CUDA program and no longer reproduces it.
