`kornia.enhance.equalize_clahe` and `kornia.augmentation.RandomClahe` now raise a `RuntimeError`
naming the `[0, 1]` input range for values the 256-bin tile lookup cannot index, instead of a raw
`index ... is out of bounds for dimension 5 with size 256` from the gather — the same named-range
treatment `equalize` got in #4489. The check uses `torch._assert_async`, so it adds no device sync
and `torch.compile` fullgraph still works; inputs the lookup can index (including values a hair
above 1) are unchanged.
