`kornia.morphology.dilation` (and `closing`, `gradient`, `top_hat`, `bottom_hat`, `opening`, which call it) now
anchors the reflected structuring element at `origin`. It used to reflect the kernel values but pad with the
unreflected origin, so an even-sized kernel (for example `torch.ones(2, 2)`) shifted the result by one pixel
toward the bottom-right relative to `scipy.ndimage.grey_dilation` and `skimage.morphology.dilation`, and a custom
`origin` moved the window in the opposite direction to the one `erosion` uses. Odd kernels with the default origin
are unchanged.
