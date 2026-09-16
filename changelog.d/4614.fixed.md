`LoFTR` fine matching returns finite gradients through `expec_f` in `float16` on torch older than 2.14 when a heatmap has zero variance; the `1e-10` floor under `sqrt` underflowed to zero there.
