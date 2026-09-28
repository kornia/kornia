`RenderingDeFMO` (used by `DeFMO`) no longer crashes on a half-precision forward pass. Its rendering
time-steps (`times`) were a plain Python attribute, not a registered buffer, so `nn.Module.to()` never
moved it; `forward` re-derived `times`'s device from the input on every call but never its dtype, so
`RenderingDeFMO().half()` left `times` at `float32` and the first forward raised `RuntimeError: Input
type (torch.FloatTensor) and weight type (torch.HalfTensor) should be the same`. `times` is now a
non-persistent buffer (it is fully determined by `tsr_steps`, not learned, so `state_dict()` keys and
existing checkpoints are unaffected), and `forward` casts into a local variable rather than depending
on the caller having matched dtype. Same bug shape as #4069/#4079 in a different class; the normal
`float32` path is byte-identical to before. (#4319)
