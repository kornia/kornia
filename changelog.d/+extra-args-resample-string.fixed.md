`AugmentationSequential` now normalizes a string or int `resample` in `extra_args` as the augmentation
constructors do. It used to raise `AttributeError` wherever the mask was resampled. (#4815)
