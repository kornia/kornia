`LoFTR` misplaced its keypoints for image sizes that are not multiples of 8. It converted the coarse cells of both
images to pixels with `H0 / ceil(H0 / 8)` instead of the 8-pixel stride of the coarse feature map, cut the
fine-matching windows of both images on a grid of the wrong stride, and scaled the fine offsets by `H0 / ceil(H0 / 2)`;
its FPN also upsampled each level to the size of its skip connection, which reads the coarser level at other positions
than the trained weights expect whenever a side is not 0 or 7 mod 8. On a 361 x 485 crop of Oxford graf and a known
homography, 4 % of the matches were within 1 px of the ground truth (median error 2.7 px). Coarse and fine cells are
now converted with the 8- and 2-pixel strides of the feature maps for any image size, `mask0` / `mask1` are read at
the pixels the coarse cells sit on, and the FPN upsamples by 2 and crops to the skip size (99 % within 1 px, median
0.23 px on the same pair; IMC PhotoTourism pose mAA@10 0.49 -> 0.72). Inputs whose heights and widths are all multiples
of 8 give the same results as before.

`LoFTR` with `mask0` / `mask1` now runs in float16. It raised a `RuntimeError`, because the similarity -1e9 given to
masked pairs does not fit in float16, and padded queries overflowed the linear attention's normaliser to inf, which
made the features nan. Masked pairs now take the lowest finite float16 value, and padded queries a zero normaliser;
float32 results are unchanged.
