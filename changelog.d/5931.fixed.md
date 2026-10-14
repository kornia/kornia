`kornia.augmentation.CenterCrop` and `kornia.augmentation.RandomResizedCrop` in `cropping_mode="slice"` raise
`NotImplementedError` from `inverse` for every image and mask batch, as `RandomCrop` already does. They checked the
mode only when inverting applied rows, so an empty batch, or one where no row was augmented, was returned instead.
Boxes and keypoints still invert in slice mode.
