`RandomCutMixV2` and `CutmixGenerator` document `cut_size` as what it is: the `[min, max]` clamp on the
Beta-sampled mixing coefficient `lambda`, where the cut side is `floor(sqrt(1 - lambda) * side)`, so a larger
`cut_size` gives a smaller cut. It was described as the "minimum and maximum cut ratio". A minimum of `1.0` is
now rejected with a `ValueError`: it forced `lambda = 1`, built an inverted zero-size box and silently made the
augmentation an identity. (#4439, #4491)
