from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import torch

from kornia.augmentation.auto.operations.base import OperationBase
from kornia.augmentation.auto.operations.policy import PolicySequential
from kornia.augmentation.container.base import ImageSequentialBase
from kornia.augmentation.container.params import ParamItem
from kornia.core import Module, Tensor
from kornia.utils import eye_like

NUMBER = Union[float, int]
OP_CONFIG = Tuple[str, NUMBER, Optional[NUMBER]]
SUBPLOLICY_CONFIG = List[OP_CONFIG]


class PolicyAugmentBase(ImageSequentialBase):
    """Policy-based image augmentation."""

    def __init__(self, policy: List[SUBPLOLICY_CONFIG]) -> None:
        policies = self.compose_policy(policy)
        super().__init__(*policies)

    def compose_policy(self, policy: List[SUBPLOLICY_CONFIG]) -> List[PolicySequential]:
        """Compose policy by the provided policy config."""
        policies = []
        for subpolicy in policy:
            policies.append(self.compose_subpolicy_sequential(subpolicy))
        return policies

    def compose_subpolicy_sequential(self, subpolicy: SUBPLOLICY_CONFIG) -> PolicySequential:
        raise NotImplementedError

    def identity_matrix(self, input: Tensor) -> Tensor:
        """Return identity matrix."""
        return eye_like(3, input)

    def get_transformation_matrix(
        self,
        input: Tensor,
        params: Optional[List[ParamItem]] = None,
        recompute: bool = False,
        extra_args: Dict[str, Any] = {},
    ) -> Optional[Tensor]:
        """Compute the transformation matrix according to the provided parameters.

        Args:
            input: the input tensor.
            params: params for the sequence.
            recompute: if to recompute the transformation matrix according to the params.
                default: False.
        """
        if params is None:
            raise NotImplementedError("requires params to be provided.")
        named_modules: Iterator[Tuple[str, Module]] = self.get_forward_sequence(params)

        # Define as 1 for broadcasting
        res_mat: Optional[Tensor] = None
        for (_, module), param in zip(named_modules, params if params is not None else []):
            module = cast(PolicySequential, module)
            mat = module.get_transformation_matrix(
                input, params=cast(Optional[List[ParamItem]], param.data), recompute=recompute, extra_args=extra_args
            )
            res_mat = mat if res_mat is None else mat @ res_mat
        return res_mat

    def is_intensity_only(self, params: Optional[List[ParamItem]] = None) -> bool:
        named_modules: Iterator[Tuple[str, Module]] = self.get_forward_sequence(params)
        for _, module in named_modules:
            module = cast(PolicySequential, module)
            if not module.is_intensity_only():
                return False
        return True

    def forward_parameters(self, batch_shape: torch.Size) -> List[ParamItem]:
        named_modules: Iterator[Tuple[str, Module]] = self.get_forward_sequence()

        params: List[ParamItem] = []
        mod_param: Union[Dict[str, Tensor], List[ParamItem]]
        for name, module in named_modules:
            module = cast(OperationBase, module)
            mod_param = module.forward_parameters(batch_shape)
            param = ParamItem(name, mod_param)
            params.append(param)
        return params
