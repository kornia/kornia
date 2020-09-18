from typing import Dict

import torch
from torch.distributions import Uniform

from kornia.augmentation import AugmentationBase2D
from ..utils import _adapted_rsampling
from .utils import POLICY_FUNCS, SubPolicy


# TODO: add fillcolor = (128, 128, 128) for each policy
ImageNetPolicy = [
    SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9),
    SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5),
    SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3),
    SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6),
    SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4),

    SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8),
    SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7),
    SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2),
    SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8),
    SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6),

    SubPolicy(0.8, "rotate", 8, 0.4, "color", 0),
    SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2),
    SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8),
    SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8),
    SubPolicy(0.6, "color", 4, 1.0, "contrast", 8),

    SubPolicy(0.8, "rotate", 8, 1.0, "color", 2),
    SubPolicy(0.8, "color", 8, 0.8, "solarize", 7),
    SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8),
    SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9),
    SubPolicy(0.4, "color", 0, 0.6, "equalize", 3),

    SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4),
    SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5),
    SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8),
    SubPolicy(0.6, "color", 4, 1.0, "contrast", 8),
    SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3)
]


CIFAR10Policy = [
    SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6),
    SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9),
    SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3),
    SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9),
    SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2),

    SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7),
    SubPolicy(0.4, "color", 3, 0.6, "brightness", 7),
    SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9),
    SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1),
    SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5),

    SubPolicy(0.7, "color", 7, 0.5, "translateX", 8),
    SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8),
    SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6),
    SubPolicy(0.9, "brightness", 6, 0.2, "color", 8),
    SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3),

    SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0),
    SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4),
    SubPolicy(0.9, "color", 9, 0.6, "equalize", 6),
    SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8),
    SubPolicy(0.1, "brightness", 3, 0.7, "color", 0),

    SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3),
    SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9),
    SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3),
    SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3),
    SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1)
]


SVHNPolicy = [
    SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3),
    SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5),
    SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6),
    SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3),
    SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3),

    SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3),
    SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5),
    SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6),
    SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1),
    SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3),

    SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3),
    SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4),
    SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6),
    SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7),
    SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4),

    SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2),
    SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8),
    SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4),
    SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3),
    SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5),

    SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7),
    SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8),
    SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3),
    SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3),
    SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5)
]


class AutoAugment(AugmentationBase2D):
    """Randomly choose one of the best 24 Sub-policies.

    AutoAugment is from the paper https://arxiv.org/abs/1805.09501.
    Code referred to: https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py

    Example:
        >>> _ = torch.manual_seed(0)
        >>> auto = AutoAugment()
        >>> input = torch.randn(2, 3, 3, 5)
        >>> auto(input)
        tensor([[[[0.8784, 0.8471, 0.7529, 0.5647, 0.8471],
                  [0.6902, 0.6902, 0.8784, 0.3137, 0.7216],
                  [0.3451, 0.2824, 0.0941, 0.2196, 0.0941]],
        <BLANKLINE>
                 [[0.7529, 0.6588, 0.3137, 0.5647, 0.7843],
                  [0.5961, 0.4392, 0.6588, 0.8471, 0.7216],
                  [0.4078, 0.8157, 0.1569, 0.3765, 0.5647]],
        <BLANKLINE>
                 [[0.9412, 0.1569, 0.3765, 0.0314, 0.5020],
                  [0.2196, 0.4392, 0.0941, 0.6275, 0.4392],
                  [0.8784, 0.7843, 0.6902, 0.0314, 0.5020]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[0.2824, 0.5333, 0.4078, 0.3137, 0.8471],
                  [0.0314, 0.1569, 0.7529, 0.5961, 0.5333],
                  [0.5961, 0.5333, 0.7216, 0.5020, 0.3765]],
        <BLANKLINE>
                 [[0.4706, 0.7529, 0.8157, 0.4392, 0.4078],
                  [0.9098, 0.0941, 0.2824, 0.5333, 0.5333],
                  [0.5333, 0.3137, 0.3765, 0.4392, 0.1569]],
        <BLANKLINE>
                 [[0.3765, 0.9098, 0.7843, 0.9725, 0.5647],
                  [0.6275, 0.6275, 0.9412, 0.5961, 0.6275],
                  [0.6902, 0.0941, 0.4392, 0.2196, 0.0627]]]])
    """
    def __init__(self, policy: str = 'imagenet') -> None:
        super(AutoAugment, self).__init__(return_transform=False, same_on_batch=True, p=1., p_batch=1.)
        if policy == 'imagenet':
            self.policies = ImageNetPolicy
        elif policy == 'cifar10':
            self.policies = CIFAR10Policy
        elif policy == 'svhn':
            self.policies = SVHNPolicy
        else:
            raise NotImplementedError(f"Policy {policy} is not implemented.")
        self._policy_idx_dist = Uniform(0, len(self.policies))

    def __repr__(self):
        return "AutoAugment Policy."

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        idx = self._policy_idx_dist.rsample((1,)).long()
        to_apply = self.policies[idx.item()].to_apply()
        return dict(
            idx=idx,
            to_apply=to_apply
        )

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.policies[params['idx'].item()](input, params['to_apply'])
