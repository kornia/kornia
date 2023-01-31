from typing import List, Optional, Tuple, Union

import torch.nn as nn

from kornia.augmentation.auto.operations import OperationBase
from kornia.core import Module, Tensor

NUMBER = Union[float, int]
OP_CONFIG = Tuple[str, NUMBER, Optional[NUMBER]]
SUBPLOLICY_CONFIG = List[OP_CONFIG]


class PolicySequential(Module):
    """Policy tuple for applying multiple operations.

    Args:
        operations: a list of operations to perform.
    """

    def __init__(self, operations: List[OperationBase]) -> None:
        super().__init__()
        self.operations = operations

    def forward(self, input: Tensor) -> Tensor:
        for op in self.operations:
            input = op(input)
        return input


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
