`ImageModule`, `ImageSequential` and the augmentation containers now convert NumPy and PIL inputs by
their dtype, and return channels-last arrays for `output_type="numpy"`. `to_tensor` used to divide
every NumPy array by 255 whatever its dtype: a float image already in `[0, 1]` shrank 255 times (0.5
became 0.00196), a `uint16` image reached 257, and a `bool` mask became 1/255. Now an integer array
is divided by the maximum of its dtype (`uint8` by 255 as before, `uint16` by 65535, `int16` by
32767), a `bool` array becomes 0 and 1, and a floating array keeps its values; `uint8` inputs
convert exactly as before. `output_type="numpy"` (and `to_numpy`) used to return the `(C, H, W)` /
`(B, C, H, W)` layout of the output tensor, which `to_tensor` read back as channels-last and
transposed; it now returns `(H, W, C)` / `(B, H, W, C)`, the layout NumPy inputs use, with the
tensor's values, so a float output fed back in converts to the same tensor. Code that expected the
old layout moves the channel axis with `np.moveaxis(out, -1, -3)`.
