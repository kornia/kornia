`dice_loss` and `DiceLoss` run faster on the CPU for float32 and float64 logits with at least 2**15 values (and class
weights, if any, of the logits' dtype): the loss no longer builds a one-hot target and full-size products of it. The
intersection takes each pixel's target-class probability, the micro average sums the (weighted) probabilities over the
classes of each pixel, and the macro average gets each class's probability mass from a batched matrix-vector product and
adds the per-class sums into (batch, class) bins. On 21-class 256 x 256 logits, forward plus backward, measured on an
i7-14700K under WSL2 (where page faults cost more than on native Linux), the default micro loss is 4.6x faster at batch
8 and 4.2x at batch 32 in eager mode, the macro loss 3.3x and 3.2x and the class-weighted micro loss 3.1x and 2.9x; with
the faults removed from both paths by `GLIBC_TUNABLES` they are 3.0x to 4.3x faster, and 1.1x under `torch.compile`.
CUDA is unchanged. Values and gradients agree with the one-hot sums to rounding, with the same NaN positions; exported,
traced and autocast calls, and smaller inputs, keep the one-hot target.
