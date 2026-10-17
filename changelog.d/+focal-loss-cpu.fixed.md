`focal_loss` and `FocalLoss` run faster on the CPU for `'mean'` and `'sum'` without class weights, on float32 and float64
logits with at least 2**19 values: the target-class log-probability comes from a custom autograd function that works
through the batch in chunks and saves only the logits and two values per pixel, instead of a full-size `log_softmax`
whose output and gradients were fresh memory faulted in on every call. On 21-class 256 x 256 logits, forward plus
backward, measured on an i7-14700K under WSL2 (where page faults cost more than on native Linux), the loss is 1.9x faster
at batch 8 and 1.8x at batch 32 in eager mode, 1.4x at both with the faults removed from both paths by `GLIBC_TUNABLES`,
and 1.5x under `torch.compile`. Values and gradients agree with `log_softmax` to rounding, with the same NaN and infinity
positions; double backward and forward-mode AD are supported, and exported, traced, autocast and functorch calls keep
`log_softmax`.
