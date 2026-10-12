`kornia.contrib.super_resolution` builders construct again. `SuperResolution` never implemented
`ModelBase`'s abstract `from_config` and defined no `__init__`, so `SmallSRBuilder.build()` and
`RRDBNetBuilder.build()` both raised `TypeError` at construction — the whole public
super-resolution entry point had been unreachable since the models refactor made `from_config`
abstract. It now has a `SuperResolutionConfig` and a `from_config` that dispatches to either
builder family, and both builders are covered by tests. Fixes #4291. (#4335)
