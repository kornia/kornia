Corrected the `kornia.morphology` docstrings: `origin` defaults to `[k_h // 2, k_w // 2]` rather than
"the center ... (rounding towards zero)", `bottom_hat` returns the bottom hat rather than the top hat, and
the kernel shape is `(k_h, k_w)` over the image's `(H, W)` axes rather than `(k_x, k_y)`.
