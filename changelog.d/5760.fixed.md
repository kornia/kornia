`DeDoDe.detect` now runs with float16 and bfloat16 weights on CPU. `DeDoDeDetector.forward` lifted the decoder context
to float32 for the interpolation and fed it back to the half-precision decoder, which raised
`RuntimeError: Input type (float) and bias type (c10::Half) should be the same`. The context is now converted back to
the input dtype after the interpolation. Results in float32 and float64 are unchanged.
