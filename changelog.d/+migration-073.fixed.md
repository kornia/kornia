`kornia.feature.LightGlue` no longer probes for `flash_attn`; the SDPA path it always used is now
the only one. `kornia.feature.lightglue.Attention.enable_flash`, the attribute that combined the
constructor argument with the probe result (and so always equalled the argument, because SDPA is
present on every supported torch), was renamed to `Attention.allow_flash`. (#4287)
