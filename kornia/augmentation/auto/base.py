from typing import List, Optional, Tuple, Union
import torch.nn as nn

from kornia.core import Module
from kornia.augmentation.auto.operations.policy import PolicySequential

NUMBER = Union[float, int]
OP_CONFIG = Tuple[str, NUMBER, Optional[NUMBER]]
SUBPLOLICY_CONFIG = List[OP_CONFIG]


class PolicyAugmentBase(Module):
    """"""

    def __init__(self, policy: List[SUBPLOLICY_CONFIG]) -> None:
        super().__init__()
        self.policies = self.compose_policy(policy)

    def compose_policy(self, policy: List[SUBPLOLICY_CONFIG]) -> nn.ModuleList:
        policies = nn.ModuleList([])
        for subpolicy in policy:
            policies.append(self.compose_subpolicy_sequential(subpolicy))
        return policies

    def compose_subpolicy_sequential(self, subpolicy: SUBPLOLICY_CONFIG) -> PolicySequential:
        raise NotImplementedError
