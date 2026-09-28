Documented the shared 2D contract of `kornia.augmentation` — the `(B, C, H, W)` float working layout
and `keepdim`, `p` / `p_batch` / `same_on_batch`, where random parameters are drawn and how `torch.manual_seed`
and `params=` replay them, what a module serializes, and what `AugmentationSequential` does with each data key
(layouts, the inclusive `xyxy_plus` boxes and integer-centre flips, nearest masks, `extra_args`, `inverse`) —
as Convention blocks on the base classes and containers, with the canonical randomness and serialization
statements on the Conventions & Pitfalls page and executable pins. The contracts distinguish sampler and
returned-parameter placement, trainable range parameters, application-time replay exceptions, padding labels,
information lost by rotated tensor boxes, mask-list gates, mixed mask dtypes, dictionary-key exceptions,
and the transplantation constructors. Scope corrections and regression tests cover mix-specific contracts,
inverse support, mask erasing/filtering/precision, nested matrices, serialization and sampler limitations;
the container, `p_batch`,
`set_rng_device_and_dtype`, `state_dict` and `B = 0` defects are tracked in dedicated issues. (#4452)
