from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import torch

import kornia.augmentation as K
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

    def __init__(self, policy: List[SUBPLOLICY_CONFIG], transformation_matrix: str = "silence") -> None:
        policies = self.compose_policy(policy)
        super().__init__(*policies)

        _valid_transformation_matrix_args = ["silence", "rigid", "skip"]
        if transformation_matrix not in _valid_transformation_matrix_args:
            raise ValueError(
                f"`transformation_matrix` has to be one of {_valid_transformation_matrix_args}. "
                f"Got {transformation_matrix}."
            )
        self._transformation_matrix_arg = transformation_matrix
        self._transform_matrix: Optional[Tensor]
        self._transform_matrices: List[Tensor] = []

    def compose_policy(self, policy: List[SUBPLOLICY_CONFIG]) -> List[PolicySequential]:
        """Compose policy by the provided policy config."""
        return [self.compose_subpolicy_sequential(subpolicy) for subpolicy in policy]

    def compose_subpolicy_sequential(self, subpolicy: SUBPLOLICY_CONFIG) -> PolicySequential:
        raise NotImplementedError

    def identity_matrix(self, input: Tensor) -> Tensor:
        """Return identity matrix."""
        return eye_like(3, input)

    @property
    def transform_matrix(self) -> Optional[Tensor]:
        # In AugmentationSequential, the parent class is accessed first.
        # So that it was None in the beginning. We hereby use lazy computation here.
        if self._transform_matrix is None and len(self._transform_matrices) != 0:
            self._transform_matrix = self._transform_matrices[0]
            for mat in self._transform_matrices[1:]:
                self._update_transform_matrix(mat)
        return self._transform_matrix

    def _update_transform_matrix_by_module(self, module: Module) -> None:
        if self._transformation_matrix_arg == "skip":
            return
        if isinstance(
            module, (K.RigidAffineAugmentationBase2D, K.RigidAffineAugmentationBase3D)
        ):
            # Passed in pointer, allows lazy transformation matrix computation
            self._transform_matrices.append(module.transform_matrix)  # type: ignore
        elif self._transformation_matrix_arg == "rigid":
            raise RuntimeError(
                f"Non-rigid module `{module}` is not supported under `rigid` computation mode. "
                "Please either update the module or change the `transformation_matrix` argument."
            )

    def _update_transform_matrix(self, transform_matrix: Tensor) -> None:
        if self._transform_matrix is None:
            self._transform_matrix = transform_matrix
        else:
            self._transform_matrix = transform_matrix @ self._transform_matrix

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
            self._update_transform_matrix_by_module(module)
            param = ParamItem(name, mod_param)
            params.append(param)
        return params
