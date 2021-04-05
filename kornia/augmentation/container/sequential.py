import torch
import torch.nn as nn

from kornia.augmentation.base import _AugmentationBase, MixAugmentationBase


class Sequential(nn.Sequential):
    """A sequential container for Kornia augmentation-specific functions.

    Args:
        *args (_AugmentationBase): a list of augmentation module.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
            input tensor.

    Example:
        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)
        >>> aug_list = Sequential(
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ... return_transform=True)
        >>> output, trans_mat = aug_list(input)
        >>> trans_mat.shape
        torch.Size([2, 3, 3])
    """

    def __init__(self, *args: _AugmentationBase, return_transform: bool = True) -> None:
        super(Sequential, self).__init__(*args)
        for aug in args:
            if isinstance(aug, MixAugmentationBase):
                raise NotImplementedError(f"MixAugmentations are not supported at this moment. Got {aug}.")
        self.return_transform = return_transform

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        transform = None
        for aug in self.children():
            input = aug(input)
            if self.return_transform:
                if transform is None:
                    transform = aug._transform_matrix
                else:
                    transform = aug._transform_matrix @ transform
        if self.return_transform:
            return input, transform
        return input
