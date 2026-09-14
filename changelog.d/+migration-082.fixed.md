`nms2d` accepts a non-square `kernel_size`. It used to raise `RuntimeError: shape '[1, 1, -1, H, W]' is
invalid for input of size ...`, because the neighbourhood kernel was built with its two extents swapped
and the padding was applied to the wrong pair of edges. (#4240, #4242)
