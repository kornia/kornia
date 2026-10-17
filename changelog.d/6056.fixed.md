Fixes in the vision backbones of `kornia.models`:

- `MobileViT` raises `ValueError` for a `mode` other than `"xxs"`, `"xs"` or `"s"`, where it raised `UnboundLocalError`.
- `TinyViT.from_config("21m", ...)` builds the `"in1k_384"` and `"in1k_512"` checkpoints, named explicitly or picked by `pretrained=True` for an `img_size` of at least 384 or 512, with the window sizes they were trained with, `[12, 12, 24, 12]` and `[16, 16, 32, 16]`, as the reference TinyViT does. It built them with the 224-px windows `[7, 7, 14, 7]`, so their 12- and 16-px attention biases were bicubic-resized to 7 px without a warning, e.g. `(6, 144)` to `(6, 49)` in stage 1. Random init and the other checkpoints keep `[7, 7, 14, 7]`.
- `TinyViT.from_config` passes `embed_dims`, `depths`, `num_heads`, `window_sizes` and `drop_path_rate` to the model in place of the variant's values. Each of them raised `TypeError: ... got multiple values for keyword argument`.
