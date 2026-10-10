`LoFTR` misplaced its keypoints when the height of `image0` was not a multiple of 8. It converted the coarse cells
of both images to pixels with `H0 / ceil(H0 / 8)` instead of the 8-pixel stride of the coarse feature map, cut the
fine-matching windows of both images on a grid of the wrong stride, and scaled the fine offsets by
`H0 / ceil(H0 / 2)`: on a 361 x 485 crop of Oxford graf and a known homography, 4 % of the matches were within 1 px
of the ground truth (median error 2.7 px). Coarse and fine cells are now converted with the 8- and 2-pixel strides
of the feature maps for any image size (99 % within 1 px, median 0.23 px on the same pair), and `mask0` / `mask1`
are read at the pixels the coarse cells sit on. Inputs whose heights and widths are multiples of 8 give the same
results as before.
