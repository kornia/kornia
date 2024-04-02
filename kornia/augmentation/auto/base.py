from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import torch

from kornia.augmentation.auto.operations.base import OperationBase
from kornia.augmentation.auto.operations.policy import PolicySequential
from kornia.augmentation.container.base import ImageSequentialBase, TransformMatrixMinIn
from kornia.augmentation.container.ops import InputSequentialOps
from kornia.augmentation.container.params import ParamItem
from kornia.core import Module, Tensor
from kornia.utils import eye_like

NUMBER = Union[float, int]
OP_CONFIG = Tuple[str, NUMBER, Optional[NUMBER]]
SUBPOLICY_CONFIG = List[OP_CONFIG]


class PolicyAugmentBase(ImageSequentialBase, TransformMatrixMinIn):
    """Policy-based image augmentation."""

    def __init__(self, policy: List[SUBPOLICY_CONFIG], transformation_matrix_mode: str = "silence") -> None:
        policies = self.compose_policy(policy)
        super().__init__(*policies)
        self._parse_transformation_matrix_mode(transformation_matrix_mode)
        self._valid_ops_for_transform_computation: Tuple[Any, ...] = (PolicySequential,)

    def _update_transform_matrix_for_valid_op(self, module: PolicySequential) -> None:  # type: ignore
        self._transform_matrices.append(module.transform_matrix)

    def clear_state(self) -> None:
        self._reset_transform_matrix_state()
        return super().clear_state()

    def compose_policy(self, policy: List[SUBPOLICY_CONFIG]) -> List[PolicySequential]:
        """Compose policy by the provided policy config."""
        return [self.compose_subpolicy_sequential(subpolicy) for subpolicy in policy]

    def compose_subpolicy_sequential(self, subpolicy: SUBPOLICY_CONFIG) -> PolicySequential:
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

    def transform_inputs(self, input: Tensor, params: List[ParamItem], extra_args: Dict[str, Any] = {}) -> Tensor:
        for param in params:
            module = self.get_submodule(param.name)
            input = InputSequentialOps.transform(input, module=module, param=param, extra_args=extra_args)
        return input

    def forward(
        self, input: Tensor, params: Optional[List[ParamItem]] = None, extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        self.clear_state()

        if params is None:
            inp = input
            _, out_shape = self.autofill_dim(inp, dim_range=(2, 4))
            params = self.forward_parameters(out_shape)

        for param in params:
            module = self.get_submodule(param.name)
            input = InputSequentialOps.transform(input, module=module, param=param, extra_args=extra_args)
            self._update_transform_matrix_by_module(module)

        self._params = params
        return input
