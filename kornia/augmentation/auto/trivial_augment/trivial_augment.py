from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

import kornia.augmentation.auto.rand_augment.ops as ops
from kornia.augmentation.auto.base import SUBPLOLICY_CONFIG, PolicyAugmentBase, PolicySequential
from kornia.core import Tensor

default_policy: List[SUBPLOLICY_CONFIG] = [
    # [("identity", 0, 1)],
    # ("auto_contrast", 0, 1),
    [("equalize", 0, 1)],
    [("rotate", -30.0, 30.0)],
    [("posterize", 0.0, 4)],
    [("solarize", 0.0, 1.0)],
    # (Color, 0.1, 1.9),
    [("contrast", 0.1, 1.9)],
    [("brightness", 0.1, 1.9)],
    [("sharpness", 0.1, 1.9)],
    [("shear_x", -0.3, 0.3)],
    [("shear_y", -0.3, 0.3)],
    [("translate_x", -0.5, 0.5)],
    [("translate_y", -0.5, 0.5)],
]


class TrivialAugment(PolicyAugmentBase):
    """Apply TrivialAugment :cite:`muller2021trivialaugment` augmentation strategies.

    Args:
        policy: candidate transformations. If None, a default candidate list will be used.
    """

    def __init__(self, policy: Optional[List[SUBPLOLICY_CONFIG]] = None) -> None:
        if policy is None:
            _policy = default_policy
        else:
            _policy = policy

        super().__init__(_policy)
        selection_weights = torch.tensor([1.0 / len(self.policies)] * len(self.policies))
        self.rand_selector = Categorical(selection_weights)

    def compose_subpolicy_sequential(self, subpolicy: SUBPLOLICY_CONFIG) -> PolicySequential:
        if len(subpolicy) != 1:
            raise RuntimeError(f"Each policy must have only one operation for TrivialAugment. Got {len(subpolicy)}.")
        name, low, high = subpolicy[0]
        return PolicySequential([getattr(ops, name)(low, high)])

    def forward(self, input: Tensor) -> Tensor:
        idx = self.rand_selector.sample()
        input = self.policies[idx](input)
        return input
