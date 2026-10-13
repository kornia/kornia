`filter2d`, `filter2d_separable` and `filter3d` now accept an empty image batch together with an empty
per-sample kernel. These inputs previously raised a `RuntimeError` about reshaping zero elements with an ambiguous
dimension. The output keeps the empty batch and the usual spatial size, including the smaller size for 2D `valid`
padding. Connections to both input and kernel gradients, integer/bool promotion to float32, and non-empty filtering
are preserved. This also fixes empty per-sample sigma in Gaussian blur and `RandomGaussianBlur`. Fixed-size motion
blur works where PyTorch supports generating the empty kernel (2D nearest/bilinear and 3D nearest); older PyTorch
constant-padding limitations, ranged kernel-size sampling and 3D bilinear warping remain outside this fix.
Part of #4429.
