`nms3d` accepts any `kernel_size`. Every size other than `(3, 3, 3)`, the one served by a hand-written
branch, used to raise: `_compute_zero_padding3d` defined a `(k - 1) // 2` helper and then returned the
full kernel sizes, so the padded volume did not match the kernel it was convolved with. (#4241, #4242)
