`kornia.core.external.transformers` was removed; it was a `LazyLoader` handle no kornia code used.
Previously `from kornia.core.external import transformers` gave a lazy proxy that imported
`transformers` on first attribute access; that name no longer exists, so import `transformers`
directly instead. (#4301)
