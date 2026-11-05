`pixi run -e cuda install` now installs the locked PyTorch as a CUDA build. It used to reinstall torch from the CUDA 12.1
index, which stops at torch 2.5.1, so a CUDA development environment silently ran the oldest supported PyTorch and
moved several locked dependencies, such as triton and sympy, to other versions. Re-run this task on a `.venv` the old one
created: `pixi run -e default install` there removes files the locked NVIDIA wheels share with the old ones, and
`import torch` then fails on `libcudnn.so.9`.
