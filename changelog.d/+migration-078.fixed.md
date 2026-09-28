`RandAugment`, `AutoAugment`, `TrivialAugment` and `AugMix` no longer silently upcast half-precision
batches to `float32`. `OperationBase.forward` — the shared gate every auto-augment op routes through —
moved its `batch_prob` mask to `input.device` but not `input.dtype`, and since `batch_prob` is `float32`
the expression `batch_prob * op(x) + (1 - batch_prob) * x` promoted a `float16` / `bfloat16` input up to
`float32` (the op output and input are both low precision, only the gate was not). `float32` / `float64`
inputs are unaffected and their numeric output is byte-identical. The mask is now cast to `input.dtype`
as well. Partially addresses #4153: this clears the `RuntimeError` behind nineteen of that issue's Linux
CPU half-precision xfail lines for `tests/augmentation/test_auto_operation.py` (all ten `cpu_bfloat16`
entries, nine of the ten `cpu_float16`), which are removed from `testing/half_precision_xfails/`. The
tenth `float16` line is kept: `TestAutoAugment::test_reproduce[cpu-float16]` still fails, now on an
`AssertionError` rather than a `RuntimeError` — a smaller residual, not a fix. (#4210)
