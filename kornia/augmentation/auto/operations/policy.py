from typing import Dict, Iterator, List, Tuple, Union, cast

from torch import Size

from kornia.augmentation.auto.operations import OperationBase
from kornia.augmentation.container.base import ParamItem
from kornia.augmentation.container.image import ImageSequentialBase
from kornia.core import Module, Tensor


class PolicySequential(ImageSequentialBase):
    """Policy tuple for applying multiple operations.

    Args:
        operations: a list of operations to perform.
    """

    def __init__(self, *operations: OperationBase) -> None:
        self.validate_operations(operations)
        super().__init__(*operations)

    def validate_operations(self, operations: List[OperationBase]) -> None:
        invalid_ops = []
        for op in operations:
            if not isinstance(op, OperationBase):
                invalid_ops.append(op)
        if len(invalid_ops) != 0:
            raise ValueError(f"All operations must be Kornia Operations. Got {invalid_ops}.")

    def forward_parameters(self, batch_shape: Size) -> List[ParamItem]:
        named_modules: Iterator[Tuple[str, Module]] = self.named_children()

        params: List[ParamItem] = []
        mod_param: Union[Dict[str, Tensor], List[ParamItem]]
        for name, module in named_modules:
            module = cast(OperationBase, module)
            mod_param = module.op.forward_parameters(batch_shape)
            param = ParamItem(name, mod_param)
            params.append(param)
        return params
