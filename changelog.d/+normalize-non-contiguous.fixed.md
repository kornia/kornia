`kornia.enhance.normalize` and `kornia.enhance.normalize_min_max` (and so `kornia.augmentation.Normalize`
and `RandomAutoContrast`) accept non-contiguous inputs, such as a tensor with transposed spatial axes,
instead of raising from `Tensor.view`.
