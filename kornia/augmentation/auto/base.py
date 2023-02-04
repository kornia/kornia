from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch

from kornia.augmentation.auto.operations.policy import PolicySequential
from kornia.augmentation.container.base import ParamItem
from kornia.augmentation.container.image import ImageSequentialBase
from kornia.core import Tensor, Module

NUMBER = Union[float, int]
OP_CONFIG = Tuple[str, NUMBER, Optional[NUMBER]]
SUBPLOLICY_CONFIG = List[OP_CONFIG]


class PolicyAugmentBase(ImageSequentialBase):
    """"""

    def __init__(self, policy: List[SUBPLOLICY_CONFIG]) -> None:
        policies = self.compose_policy(policy)
        super().__init__(*policies)

    def compose_policy(self, policy: List[SUBPLOLICY_CONFIG]) -> List[PolicySequential]:
        policies = []
        for subpolicy in policy:
            policies.append(self.compose_subpolicy_sequential(subpolicy))
        return policies

    def compose_subpolicy_sequential(self, subpolicy: SUBPLOLICY_CONFIG) -> PolicySequential:
        raise NotImplementedError

    def forward_parameters(self, batch_shape: torch.Size) -> List[ParamItem]:
        named_modules: Iterator[Tuple[str, Module]] = self.get_forward_sequence()

        params: List[ParamItem] = []
        mod_param: Union[Dict[str, Tensor], List[ParamItem]]
        for name, module in named_modules:
            mod_param = module.forward_parameters(batch_shape)
            param = ParamItem(name, mod_param)
            params.append(param)

        return params
