from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

from torch import Size

import kornia.augmentation as K
from kornia.augmentation.auto.operations import OperationBase
from kornia.augmentation.container.base import ImageSequentialBase
from kornia.augmentation.utils import override_parameters
from kornia.core import Module, Tensor, as_tensor
from kornia.utils import eye_like


class PolicySequential(ImageSequentialBase):
    """Policy tuple for applying multiple operations.

    Args:
        operations: a list of operations to perform.
    """

    def __init__(self, *operations: OperationBase) -> None:
        self.validate_operations(*operations)
        super().__init__(*operations)

    def validate_operations(self, *operations: OperationBase) -> None:
        invalid_ops: List[OperationBase] = []
        for op in operations:
            if not isinstance(op, OperationBase):
                invalid_ops.append(op)
        if len(invalid_ops) != 0:
            raise ValueError(f"All operations must be Kornia Operations. Got {invalid_ops}.")

    def identity_matrix(self, input: Tensor) -> Tensor:
        """Return identity matrix."""
        return eye_like(3, input)

    def get_transformation_matrix(
        self,
        input: Tensor,
        params: Optional[List[K.container.ParamItem]] = None,
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
            module = cast(OperationBase, module)
            if isinstance(module.op, (K.GeometricAugmentationBase2D,)) and isinstance(param.data, dict):
                to_apply = param.data['batch_prob']
                ori_shape = input.shape
                input = module.op.transform_tensor(input)
                # Standardize shape
                if recompute:
                    mat: Tensor = self.identity_matrix(input)
                    flags = override_parameters(module.op.flags, extra_args, in_place=False)
                    mat[to_apply] = module.op.compute_transformation(input[to_apply], param.data, flags)
                else:
                    mat = as_tensor(module.op._transform_matrix, device=input.device, dtype=input.dtype)
                res_mat = mat if res_mat is None else mat @ res_mat
                input = module.op.transform_output_tensor(input, ori_shape)
                if module.op.keepdim and ori_shape != input.shape:
                    res_mat = res_mat.squeeze()
        return res_mat

    def is_intensity_only(self) -> bool:
        for module in self.children():
            module = cast(OperationBase, module)
            if isinstance(module.op, (K.GeometricAugmentationBase2D,)):
                return False
        return True

    def get_forward_sequence(
        self, params: Optional[List[K.container.ParamItem]] = None
    ) -> Iterator[Tuple[str, Module]]:
        if params is not None:
            return super().get_children_by_params(params)
        return self.named_children()

    def forward_parameters(self, batch_shape: Size) -> List[K.container.ParamItem]:
        named_modules: Iterator[Tuple[str, Module]] = self.get_forward_sequence()

        params: List[K.container.ParamItem] = []
        mod_param: Union[Dict[str, Tensor], List[K.container.ParamItem]]
        for name, module in named_modules:
            module = cast(OperationBase, module)
            mod_param = module.op.forward_parameters(batch_shape)
            param = K.container.ParamItem(name, mod_param)
            params.append(param)
        return params
