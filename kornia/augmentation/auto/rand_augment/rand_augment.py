from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from kornia.augmentation.auto.base import SUBPLOLICY_CONFIG, PolicyAugmentBase
from kornia.augmentation.auto.operations import OperationBase
from kornia.core import Tensor

from . import ops

default_policy: List[SUBPLOLICY_CONFIG] = [
    # ("auto_contrast", 0, 1),
    [("equalize", 0, 1)],
    [("invert", 0, 1)],
    [("rotate", -30.0, 30.0)],
    [("posterize", 0.0, 4)],
    [("solarize", 0.0, 1.0)],
    [("solarize_add", 0.0, 0.43)],
    [("color", 0.1, 1.9)],
    [("contrast", 0.1, 1.9)],
    [("brightness", 0.1, 1.9)],
    [("sharpness", 0.1, 1.9)],
    [("shear_x", -0.3, 0.3)],
    [("shear_y", -0.3, 0.3)],
    # (CutoutAbs, 0, 40),
    [("translate_x", -0.1, 0.1)],
    [("translate_x", -0.1, 0.1)],
]


class RandAugment(PolicyAugmentBase):
    """Apply RandAugment :cite:`cubuk2020randaugment` augmentation strategies.

    Args:
        n: the number of augmentations to apply sequentially.
        m: magnitude for all the augmentations, ranged from [0, 30].
        policy: candidate transformations. If None, a default candidate list will be used.
    """

    def __init__(self, n: int, m: int, policy: Optional[List[SUBPLOLICY_CONFIG]] = None) -> None:
        if m <= 0 or m >= 30:
            raise ValueError(f"Expect `m` in [0, 30]. Got {m}.")

        if policy is None:
            _policy = default_policy
        else:
            _policy = policy

        super().__init__(_policy)
        selection_weights = torch.tensor([1.0 / len(self.policies)] * len(self.policies))
        self.rand_selector = Categorical(selection_weights)
        self.n = n
        self.m = m

    def compose_policy(self, policy: List[SUBPLOLICY_CONFIG]) -> nn.ModuleList:
        """Obtain the policies according to the policy JSON."""

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
