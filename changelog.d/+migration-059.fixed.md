`_cdist` replaces `dm.clamp(min=0.0).sqrt()` with a masked argument substitution, avoiding `NaN` gradients for
exact zero-distance pairs on PyTorch 2.5.1 and 2.9.1 where `clamp`'s boundary gradient passed through to `sqrt(0)`.
PyTorch 2.14.0 was already finite, and forward behavior and computed distances are unchanged. (#4233)
