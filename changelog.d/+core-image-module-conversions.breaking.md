`ImageModule`, `ImageSequential` and the augmentation containers now convert NumPy, PIL and
image-path inputs by their dtype, and return channels-last arrays for `output_type="numpy"`.
`to_tensor` used to divide every input by 255 whatever its dtype: a float image already in `[0, 1]`
shrank 255 times (0.5 became 0.00196), a `uint16` image or 16-bit PNG reached 257, and a `bool` mask
became 1/255. Now an integer input is divided by the maximum of its dtype (`uint8` by 255 as before,
`uint16` by 65535, `int16` by 32767), a `bool` array becomes 0 and 1, and a floating array keeps its
values; `uint8` inputs convert exactly as before. A signed integer image maps to
`[iinfo.min / iinfo.max, 1]`, so its negative values stay negative (`int8` -128 gives -128 / 127).
`output_type="numpy"` (and `to_numpy`) used to return the `(C, H, W)` / `(B, C, H, W)` layout of
the output tensor, which `to_tensor` read back as channels-last and transposed; it now returns
`(H, W, C)` / `(B, H, W, C)`, the layout NumPy inputs use, with the tensor's values, so a float
output fed back in converts to the same tensor. Every 3-D or 4-D output is treated as an image, so a
non-image element of a tuple output (such as a `(B, N, 4)` box tensor) moves its axis too; modules
with several outputs are #5210. Code that expected the old layout moves the channel axis with
`np.moveaxis(out, -1, -3)`.
