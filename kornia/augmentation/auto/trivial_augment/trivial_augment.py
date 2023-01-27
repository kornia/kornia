from typing import Optional, List
import torch
import torch.nn as nn
from torch.distributions import Categorical

import kornia.augmentation.auto.rand_augment.ops as ops
from kornia.augmentation.auto.operations import OperationBase
from kornia.augmentation.auto.autoaugment.autoaugment import SUBPLOLICY_CONFIG
from kornia.core import Tensor, Module

default_policy = [
    # [("identity", 0, 1)],
    # ("auto_contrast", 0, 1),
    [("equalize", 0, 1)],
    [("rotate", -30., 30.)],
    [("posterize", 0., 4)],
    [("solarize", 0., 1.)],
    # (Color, 0.1, 1.9),
    [("contrast", 0.1, 1.9)],
    [("brightness", 0.1, 1.9)],
    [("sharpness", 0.1, 1.9)],
    # (ShearX, 0., 0.3),
    # (ShearY, 0., 0.3),
    # (TranslateXabs, 0., 100),
    # (TranslateYabs, 0., 100),
]


class TrivialAugment(Module):
    """Apply TrivialAugment :cite:`muller2021trivialaugment` augmentation strategies.

    Args:
        policy: candidate transformations. If None, a default candidate list will be used.
    """

    def __init__(self, policy: Optional[List[SUBPLOLICY_CONFIG]] = None) -> None:
        super().__init__()

        if policy is None:
            _policy = default_policy
        else:
            _policy = policy

        self.policies = self.compose_policy(_policy)
        selection_weights = torch.tensor([1. / len(self.policies)] * len(self.policies))
        self.rand_selector = Categorical(selection_weights)

    def compose_policy(self, policy: List[OperationBase]) -> nn.ModuleList:
        """Obtain the policies according to the policy JSON."""

        def _get_op(subpolicy: SUBPLOLICY_CONFIG) -> OperationBase:
            name, low, high = subpolicy[0]
            return getattr(ops, name)(low, high)

        policies = nn.ModuleList([])
        for subpolicy in policy:
            policies.append(_get_op(subpolicy))
        return policies

    def forward(self, input: Tensor) -> Tensor:
        idx = self.rand_selector.sample()
        input = self.policies[idx](input)
        return input
