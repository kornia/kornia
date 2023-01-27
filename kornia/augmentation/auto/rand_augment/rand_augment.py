from typing import Optional, List
import torch
import torch.nn as nn
from torch.distributions import Categorical

from kornia.augmentation.auto.operations import OperationBase
from kornia.augmentation.auto.autoaugment.autoaugment import SUBPLOLICY_CONFIG
from kornia.core import Tensor, Module
from . import ops

default_policy = [
    # ("auto_contrast", 0, 1),
    [("equalize", 0, 1)],
    [("invert", 0, 1)],
    [("rotate", -30., 30.)],
    [("posterize", 0., 4)],
    [("solarize", 0., 1.)],
    [("solarize_add", 0., .43)],
    # (Color, 0.1, 1.9),
    [("contrast", 0.1, 1.9)],
    [("brightness", 0.1, 1.9)],
    [("sharpness", 0.1, 1.9)],
    # (ShearX, 0., 0.3),
    # (ShearY, 0., 0.3),
    # (CutoutAbs, 0, 40),
    # (TranslateXabs, 0., 100),
    # (TranslateYabs, 0., 100),
]


class RandAugment(Module):
    def __init__(self, n: int, m: int, policy: Optional[List[SUBPLOLICY_CONFIG]] = None) -> None:
        super().__init__()

        if m <= 0 or m >= 30:
            raise ValueError(f"Expect `m` in [0, 30]. Got {m}.")

        if policy is None:
            _policy = default_policy
        else:
            _policy = policy

        self.policies = self.compose_policy(_policy)
        selection_weights = torch.tensor([1. / len(self.policies)] * len(self.policies))
        self.rand_selector = Categorical(selection_weights)
        self.n = n
        self.m = m

    def compose_policy(self, policy: List[OperationBase]) -> nn.ModuleList:

        def _get_op(subpolicy: SUBPLOLICY_CONFIG) -> OperationBase:
            name, low, high = subpolicy[0]
            return getattr(ops, name)(low, high)

        policies = nn.ModuleList([])
        for subpolicy in policy:
            policies.append(_get_op(subpolicy))
        return policies

    def forward(self, input: Tensor) -> Tensor:
        indices = self.rand_selector.sample((self.n,))
        batch_size = input.size(0)

        m = torch.tensor([self.m / 30] * batch_size)

        for idx in indices:
            op: OperationBase = self.policies[idx]
            mag = None
            if op.magnitude_range is not None:
                minval, maxval = op.magnitude_range
                mag = m * float(maxval - minval) + minval
            input = op(input, mag=mag)

        return input
