`LoFTR.forward` is about 1.3x faster and allocates about 30 % less peak GPU memory: on 1024 px image pairs (IMC 2021
PhotoTourism validation, RTX 4090, float32) it takes 64 ms instead of 84 ms per pair, and the coarse matching keeps two
similarity-sized matrices alive instead of four. The fine-matching windows are cut only at the matched coarse cells,
coarse matches are selected from the row and column maxima of the confidence matrix instead of from boolean masks over
the whole matrix, the dual softmax no longer uses PyTorch's slow `softmax(dim=1)` kernel (and runs in place when no
gradient is needed), and linear attention over sequences longer than its channel count runs as dense GEMMs. Outputs
change at the rounding level only: on the 11,707 validation pairs, 105 of 19.8 million coarse matches differ, and pose
accuracy is unchanged. The match selection itself differs from before only when a row of the confidence matrix has two
exactly equal maxima.
