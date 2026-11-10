`match_nn`, `match_mnn`, `match_snn` and `match_smnn` select their matches on squared distances, the matrix product
that `torch.cdist` computes before its square root, and take the square root of the selected distances only. On CUDA,
the ratio tests of `match_snn` and `match_smnn` find the two nearest neighbours with two minimum reductions instead of
`torch.topk`. The returned distances and ratios are bit for bit those of `torch.cdist`. On an RTX 4090 with 4,096 to
8,192 RootSIFT descriptors per image, the four are 1.2-2.6x faster. On an x86 i7-14700K they are 1.1-1.4x faster from
1,024 descriptors on. Two distances that differ only below the precision of their square roots are no longer tied, and
the nearer one is returned. The matchers keep `torch.cdist` in these cases: inputs that need gradients, half precision,
autocast, exported or traced calls, MPS, and matrices smaller than 2,048 x 2,048 on CUDA or 256 x 256 on the CPU.
