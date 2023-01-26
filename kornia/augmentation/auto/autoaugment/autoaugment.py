from typing import List, Union

import torch
import torch.nn as nn
from torch.distributions import Categorical

from kornia.augmentation.auto.operations import OperationBase
from kornia.core import Tensor, Module
from .config import (
    SUBPLOLICY_CONFIG,
    imagenet_policy,
    cifar10_policy,
    svhn_policy
)
from . import ops


class SubPolicy(Module):
    def __init__(self, operations: List[OperationBase]) -> None:
        super().__init__()
        self.operations = operations

    def forward(self, input: Tensor) -> Tensor:
        for op in self.operations:
            input = op(input)
        return input


class AutoAugment(Module):
    def __init__(self, policy: Union[str, List[SUBPLOLICY_CONFIG]] = "imagenet") -> None:
        super().__init__()
        if policy == "imagenet":
            _policy = imagenet_policy
        elif policy == "cifar10":
            _policy = cifar10_policy
        elif policy == "svhn":
            _policy = svhn_policy
        elif isinstance(policy, (list, tuple,)):
            _policy = policy
        else:
            raise NotImplementedError(f"Invalid policy `{policy}`.")
        
        self.policies = self.compose_policy(_policy)
        selection_weights = torch.tensor([1. / len(self.policies)] * len(self.policies))
        self.rand_selector = Categorical(selection_weights)

    def compose_policy(self, policy: List[SUBPLOLICY_CONFIG]) -> nn.ModuleList:

        def _get_subpolicy(subpolicy: SUBPLOLICY_CONFIG) -> SubPolicy:
            return SubPolicy([getattr(ops, name)(prob, mag) for name, prob, mag in subpolicy])

        policies = nn.ModuleList([])
        for subpolicy in policy:
            policies.append(_get_subpolicy(subpolicy))
        return policies

    def forward(self, input: Tensor) -> Tensor:
        idx = self.rand_selector.sample()
        return self.policies[idx](input)
