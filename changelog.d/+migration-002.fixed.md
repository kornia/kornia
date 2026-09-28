The `Args` blocks of `ColorJitter`, `RandomBrightness` and `RandomGaussianBlur` no longer document a
`silence_instantiation_warning` argument that none of them accepts, and `ColorJitter` now documents its
`order` argument: a fixed (sub)set of brightness/contrast/saturation/hue indices that makes the transform
`torch.compile` fullgraph-safe, with the drawn `_params["order"]` entry ignored when it is set.
(#4437, #4490)
