With a channels_last feature map, `ALIKED.extract_dense_map` now normalises each pixel's channels in an explicit
(B, H, W, C) tensor. #6498 made `ALIKED.forward` ask for that map on the CPU, and under `torch.compile` with static
shapes inductor then transposed every tile of the map into its store. That made the compiled forward of a 416 x 640
image on an x86 i7-14700K 1.28x slower: 254 ms with the map against 199 ms without. The new form gives the same eager
bits and takes the compiled forward to 228 ms. The rest of the cost is in inductor's `grid_sample` in the descriptor
head. Eager, CUDA and MPS are unchanged.
