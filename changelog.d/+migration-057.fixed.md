`RandomChannelDropout.fill_value` and `RandomGaussianBlurGenerator.sigma` (when passed as a
Tensor) are registered as non-persistent buffers instead of plain attributes, so they move with
`Module.to()`/`.half()`/`.cuda()` and appear in `named_buffers()` like the rest of the module's
state; `state_dict()` keys are unchanged. Neither crashed before this change (both re-derived
device and dtype inline on every call), so this is a hygiene fix, not a bug fix for a crash.
One user-visible effect: after `.half()`/`.bfloat16()` the fill value is now held in that dtype,
so an input already in the module's dtype is unaffected, while a `float32` input fed to a
half-converted `RandomChannelDropout(fill_value=0.3)` is filled with `0.300048828125` (float16)
or `0.30078125` (bfloat16) rather than `0.3` -- the same buffer semantics as `Normalize`.
(#4337)
