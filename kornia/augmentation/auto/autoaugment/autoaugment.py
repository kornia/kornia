from typing import Iterator, List, Optional, Tuple, Union

from torch.distributions import Categorical

from kornia.augmentation.auto.base import SUBPOLICY_CONFIG, PolicyAugmentBase
from kornia.augmentation.auto.operations.policy import PolicySequential
from kornia.augmentation.container.params import ParamItem
from kornia.core import Module, tensor

from . import ops

imagenet_policy: List[SUBPOLICY_CONFIG] = [
    [("posterize", 0.4, 8), ("rotate", 0.6, 9)],
    [("solarize", 0.6, 5), ("auto_contrast", 0.6, None)],
    [("equalize", 0.8, None), ("equalize", 0.6, None)],
    [("posterize", 0.6, 7), ("posterize", 0.6, 6)],
    [("equalize", 0.4, None), ("solarize", 0.2, 4)],
    [("equalize", 0.4, None), ("rotate", 0.8, 8)],
    [("solarize", 0.6, 3), ("equalize", 0.6, None)],
    [("posterize", 0.8, 5), ("equalize", 1.0, None)],
    [("rotate", 0.2, 3), ("solarize", 0.6, 8)],
    [("equalize", 0.6, None), ("posterize", 0.4, 6)],
    [("rotate", 0.8, 8), ("color", 0.4, 0)],
    [("rotate", 0.4, 9), ("equalize", 0.6, None)],
    [("equalize", 0.0, None), ("equalize", 0.8, None)],
    [("invert", 0.6, None), ("equalize", 1.0, None)],
    [("color", 0.6, 4), ("contrast", 1.0, 8)],
    [("rotate", 0.8, 8), ("color", 1.0, 2)],
    [("color", 0.8, 8), ("solarize", 0.8, 7)],
    [("sharpness", 0.4, 7), ("invert", 0.6, None)],
    [("shear_x", 0.6, 5), ("equalize", 1.0, None)],
    [("color", 0.4, 0), ("equalize", 0.6, None)],
    [("equalize", 0.4, None), ("solarize", 0.2, 4)],
    [("solarize", 0.6, 5), ("auto_contrast", 0.6, None)],
    [("invert", 0.6, None), ("equalize", 1.0, None)],
    [("color", 0.6, 4), ("contrast", 1.0, 8)],
    [("equalize", 0.8, None), ("equalize", 0.6, None)],
]


cifar10_policy: List[SUBPOLICY_CONFIG] = [
    [("invert", 0.1, None), ("contrast", 0.2, 6)],
    [("rotate", 0.7, 2), ("translate_x", 0.3, 9)],
    [("sharpness", 0.8, 1), ("sharpness", 0.9, 3)],
    [("shear_y", 0.5, 8), ("translate_y", 0.7, 9)],
    [("auto_contrast", 0.5, None), ("equalize", 0.9, None)],
    [("shear_y", 0.2, 7), ("posterize", 0.3, 7)],
    [("color", 0.4, 3), ("brightness", 0.6, 7)],
    [("sharpness", 0.3, 9), ("brightness", 0.7, 9)],
    [("equalize", 0.6, None), ("equalize", 0.5, None)],
    [("contrast", 0.6, 7), ("sharpness", 0.6, 5)],
    [("color", 0.7, 7), ("translate_x", 0.5, 8)],
    [("equalize", 0.3, None), ("auto_contrast", 0.4, None)],
    [("translate_y", 0.4, 3), ("sharpness", 0.2, 6)],
    [("brightness", 0.9, 6), ("color", 0.2, 8)],
    [("solarize", 0.5, 2), ("invert", 0.0, None)],
    [("equalize", 0.2, None), ("auto_contrast", 0.6, None)],
    [("equalize", 0.2, None), ("equalize", 0.6, None)],
    [("color", 0.9, 9), ("equalize", 0.6, None)],
    [("auto_contrast", 0.8, None), ("solarize", 0.2, 8)],
    [("brightness", 0.1, 3), ("color", 0.7, 0)],
    [("solarize", 0.4, 5), ("auto_contrast", 0.9, None)],
    [("translate_y", 0.9, 9), ("translate_y", 0.7, 9)],
    [("auto_contrast", 0.9, None), ("solarize", 0.8, 3)],
    [("equalize", 0.8, None), ("invert", 0.1, None)],
    [("translate_y", 0.7, 9), ("auto_contrast", 0.9, None)],
]


svhn_policy: List[SUBPOLICY_CONFIG] = [
    [("shear_x", 0.9, 4), ("invert", 0.2, None)],
    [("shear_y", 0.9, 8), ("invert", 0.7, None)],
    [("equalize", 0.6, None), ("solarize", 0.6, 6)],
    [("invert", 0.9, None), ("equalize", 0.6, None)],
    [("equalize", 0.6, None), ("rotate", 0.9, 3)],
    [("shear_x", 0.9, 4), ("auto_contrast", 0.8, None)],
    [("shear_y", 0.9, 8), ("invert", 0.4, None)],
    [("shear_y", 0.9, 5), ("solarize", 0.2, 6)],
    [("invert", 0.9, None), ("auto_contrast", 0.8, None)],
    [("equalize", 0.6, None), ("rotate", 0.9, 3)],
    [("shear_x", 0.9, 4), ("solarize", 0.3, 3)],
    [("shear_y", 0.8, 8), ("invert", 0.7, None)],
    [("equalize", 0.9, None), ("translate_y", 0.6, 6)],
    [("invert", 0.9, None), ("equalize", 0.6, None)],
    [("contrast", 0.3, 3), ("rotate", 0.8, 4)],
    [("invert", 0.8, None), ("translate_y", 0.0, 2)],
    [("shear_y", 0.7, 6), ("solarize", 0.4, 8)],
    [("invert", 0.6, None), ("rotate", 0.8, 4)],
    [("shear_y", 0.3, 7), ("translate_x", 0.9, 3)],
    [("shear_x", 0.1, 6), ("invert", 0.6, None)],
    [("solarize", 0.7, 2), ("translate_y", 0.6, 7)],
    [("shear_y", 0.8, 4), ("invert", 0.8, None)],
    [("shear_x", 0.7, 9), ("translate_y", 0.8, 3)],
    [("shear_y", 0.8, 5), ("auto_contrast", 0.7, None)],
    [("shear_x", 0.7, 2), ("invert", 0.1, None)],
]


class AutoAugment(PolicyAugmentBase):
    """Apply AutoAugment :cite:`cubuk2018autoaugment` searched strategies.

    Args:
        policy: a customized policy config or presets of "imagenet", "cifar10", and "svhn".
        transformation_matrix_mode: computation mode for the chained transformation matrix, via `.transform_matrix`
                                    attribute.
                                    If `silent`, transformation matrix will be computed silently and the non-rigid
                                    modules will be ignored as identity transformations.
                                    If `rigid`, transformation matrix will be computed silently and the non-rigid
                                    modules will trigger errors.
                                    If `skip`, transformation matrix will be totally ignored.

    Examples:
        >>> import torch
        >>> import kornia.augmentation as K
        >>> in_tensor = torch.rand(5, 3, 30, 30)
        >>> aug = K.AugmentationSequential(AutoAugment())
        >>> aug(in_tensor).shape
        torch.Size([5, 3, 30, 30])
    """

    def __init__(
        self, policy: Union[str, List[SUBPOLICY_CONFIG]] = "imagenet", transformation_matrix_mode: str = "silent"
    ) -> None:
        if policy == "imagenet":
            _policy = imagenet_policy
        elif policy == "cifar10":
            _policy = cifar10_policy
        elif policy == "svhn":
            _policy = svhn_policy
        elif isinstance(policy, (list, tuple)):
            _policy = policy
        else:
            raise NotImplementedError(f"Invalid policy `{policy}`.")

        super().__init__(_policy, transformation_matrix_mode=transformation_matrix_mode)
        selection_weights = tensor([1.0 / len(self)] * len(self))
        self.rand_selector = Categorical(selection_weights)

    def compose_subpolicy_sequential(self, subpolicy: SUBPOLICY_CONFIG) -> PolicySequential:
        return PolicySequential(*[getattr(ops, name)(prob, mag) for name, prob, mag in subpolicy])

    def get_forward_sequence(self, params: Optional[List[ParamItem]] = None) -> Iterator[Tuple[str, Module]]:
        if params is None:
            idx = self.rand_selector.sample((1,))
            return self.get_children_by_indices(idx)

        return self.get_children_by_params(params)
