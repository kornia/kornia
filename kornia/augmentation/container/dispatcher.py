from typing import List, Tuple, Union

from torch import Tensor, nn

from .augment import AugmentationSequential


class ManyToManyAugmentationDispather(nn.Module):
    r"""Dispatches different augmentations to different inputs element-wisely.

    Args:
        augmentations: a list or a sequence of kornia AugmentationSequential modules.

    Examples:
        >>> import torch
        >>> input_1, input_2 = torch.randn(2, 3, 5, 6), torch.randn(2, 3, 5, 6)
        >>> mask_1, mask_2 = torch.ones(2, 3, 5, 6), torch.ones(2, 3, 5, 6)
        >>> aug_list = ManyToManyAugmentationDispather(
        ...     AugmentationSequential(
        ...         kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...         kornia.augmentation.RandomAffine(360, p=1.0),
        ...         data_keys=["input", "mask",],
        ...     ),
        ...     AugmentationSequential(
        ...         kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...         kornia.augmentation.RandomAffine(360, p=1.0),
        ...         data_keys=["input", "mask",],
        ...     )
        ... )
        >>> output = aug_list((input_1, mask_1), (input_2, mask_2))
    """

    def __init__(self, *augmentations: AugmentationSequential) -> None:
        super().__init__()
        self._check_consistency(*augmentations)
        self.augmentations = augmentations

    def _check_consistency(self, *augmentations: AugmentationSequential) -> bool:
        for i, aug in enumerate(augmentations):
            if not isinstance(aug, AugmentationSequential):
                raise ValueError(f"Please wrap your augmentations[`{i}`] with `AugmentationSequentials`.")
        return True

    def forward(self, *input: Union[List[Tensor], List[Tuple[Tensor]]]) -> Union[List[Tensor], List[Tuple[Tensor]]]:
        return [aug(*inp) for inp, aug in zip(input, self.augmentations)]


class ManyToOneAugmentationDispather(nn.Module):
    r"""Dispatches different augmentations to a single input and returns a list.

    Same `datakeys` must be applied across different augmentations. By default, with input
    (image, mask), the augmentations must not mess it as (mask, image) to avoid unexpected
    errors. This check can be cancelled with `strict=False` if needed.

    Args:
        augmentations: a list or a sequence of kornia AugmentationSequential modules.

    Examples:
        >>> import torch
        >>> input = torch.randn(2, 3, 5, 6)
        >>> mask = torch.ones(2, 3, 5, 6)
        >>> aug_list = ManyToOneAugmentationDispather(
        ...     AugmentationSequential(
        ...         kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...         kornia.augmentation.RandomAffine(360, p=1.0),
        ...         data_keys=["input", "mask",],
        ...     ),
        ...     AugmentationSequential(
        ...         kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...         kornia.augmentation.RandomAffine(360, p=1.0),
        ...         data_keys=["input", "mask",],
        ...     )
        ... )
        >>> output = aug_list(input, mask)
    """

    def __init__(self, *augmentations: AugmentationSequential, strict: bool = True) -> None:
        super().__init__()
        self.strict = strict
        self._check_consistency(*augmentations)
        self.augmentations = augmentations

    def _check_consistency(self, *augmentations: AugmentationSequential) -> bool:
        for i, aug in enumerate(augmentations):
            if not isinstance(aug, AugmentationSequential):
                raise ValueError(f"Please wrap your augmentations[`{i}`] with `AugmentationSequentials`.")
            if self.strict and i != 0 and aug.data_keys != augmentations[i - 1].data_keys:
                raise RuntimeError(
                    f"Different `data_keys` between {i - 1} and {i} elements, "
                    f"got {aug.data_keys} and {augmentations[i - 1].data_keys}."
                )
        return True

    def forward(self, *input: Union[Tensor, Tuple[Tensor]]) -> Union[List[Tensor], List[Tuple[Tensor]]]:
        return [aug(*input) for aug in self.augmentations]
