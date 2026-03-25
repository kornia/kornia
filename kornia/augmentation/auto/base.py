# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import torch
from torch import nn

from kornia.augmentation.auto.operations.base import OperationBase
from kornia.augmentation.auto.operations.policy import PolicySequential
from kornia.augmentation.container.base import ImageSequentialBase, TransformMatrixMinIn
from kornia.augmentation.container.ops import InputSequentialOps
from kornia.augmentation.container.params import ParamItem
from kornia.core.ops import eye_like

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
        """Clears the internal state of the augmentation, including the transform matrix."""
        self._reset_transform_matrix_state()
        return super().clear_state()

    def compose_policy(self, policy: List[SUBPOLICY_CONFIG]) -> List[PolicySequential]:
        """Compose policy by the provided policy config."""
        return [self.compose_subpolicy_sequential(subpolicy) for subpolicy in policy]

    def compose_subpolicy_sequential(self, subpolicy: SUBPOLICY_CONFIG) -> PolicySequential:
        """Composes a sequential policy module from a subpolicy configuration.

        Args:
            subpolicy: A list of operations containing their names, probabilities, and magnitudes.

        Returns:
            The composed sequential policy module.
        """
        raise NotImplementedError

    def identity_matrix(self, input: torch.Tensor) -> torch.Tensor:
        """Return identity matrix."""
        return eye_like(3, input)

    def get_transformation_matrix(
        self,
        input: torch.Tensor,
        params: Optional[List[ParamItem]] = None,
        recompute: bool = False,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        """Compute the transformation matrix according to the provided parameters.

        Args:
            input: the input torch.Tensor.
            params: params for the sequence.
            recompute: if to recompute the transformation matrix according to the params.
                default: False.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.

        """
        if params is None:
            raise NotImplementedError("requires params to be provided.")
        named_modules: Iterator[Tuple[str, nn.Module]] = self.get_forward_sequence(params)

        # Define as 1 for broadcasting
        res_mat: Optional[torch.Tensor] = None
        for (_, module), param in zip(named_modules, params if params is not None else []):
            module = cast(PolicySequential, module)
            mat = module.get_transformation_matrix(
                input, params=cast(Optional[List[ParamItem]], param.data), recompute=recompute, extra_args=extra_args
            )
            res_mat = mat if res_mat is None else mat @ res_mat
        return res_mat

    def is_intensity_only(self, params: Optional[List[ParamItem]] = None) -> bool:
        """Checks if the sequence of operations contains only intensity transformations.

        Args:
            params: Optional parameters to evaluate specific modules.

        Returns:
            True if only intensity transformations are present, False otherwise.
        """
        named_modules: Iterator[Tuple[str, nn.Module]] = self.get_forward_sequence(params)
        for _, module in named_modules:
            module = cast(PolicySequential, module)
            if not module.is_intensity_only():
                return False
        return True

    def forward_parameters(self, batch_shape: torch.Size) -> List[ParamItem]:
        """Generates the parameters for the forward pass based on the batch shape.

        Args:
            batch_shape: The shape of the input batch.

        Returns:
            A list of generated parameters for each module in the sequence.
        """
        named_modules: Iterator[Tuple[str, nn.Module]] = self.get_forward_sequence()

        params: List[ParamItem] = []
        mod_param: Union[Dict[str, torch.Tensor], List[ParamItem]]
        for name, module in named_modules:
            module = cast(OperationBase, module)
            mod_param = module.forward_parameters(batch_shape)
            param = ParamItem(name, mod_param)
            params.append(param)
        return params

    def transform_inputs(
        self, input: torch.Tensor, params: List[ParamItem], extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Transforms the input tensor using the provided parameters.

        Args:
            input: The input tensor to transform.
            params: The list of parameters for each operation.
            extra_args: Optional dictionary of extra arguments.

        Returns:
            The transformed input tensor.
        """
        for param in params:
            module = self.get_submodule(param.name)
            input = InputSequentialOps.transform(input, module=module, param=param, extra_args=extra_args)
        return input

    def forward(
        self, input: torch.Tensor, params: Optional[List[ParamItem]] = None, extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Performs the forward pass of the policy augmentation.

        Args:
            input: The input tensor.
            params: Optional parameters to use for the transformations.
            extra_args: Optional dictionary of extra arguments.

        Returns:
            The augmented input tensor.
        """
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
