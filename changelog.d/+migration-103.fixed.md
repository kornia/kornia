Preserve empty color tensors on accelerator backends instead of asking `reshape` to infer an ambiguous batch
dimension. This lets the YUV420 and YUV422 empty-input conversions return their documented empty RGB output on
MPS, matching CPU behavior (#4185).
