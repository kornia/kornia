from typing import Union

from kornia.core import ImageModule as Module
from kornia.core import Tensor, tensor


class Rescale(Module):
    r"""Initialize the Rescale operator.

    Args:
        factor: The scaling factor. Could be a float or a 0-d tensor.
    """

    def __init__(self, factor: Union[float, Tensor]) -> None:
        super().__init__()
        if isinstance(factor, float):
            self.factor = tensor(factor)
        else:
            if not isinstance(factor, Tensor) or factor.ndim != 0:
                raise TypeError(f"Expected factor to be a float or a 0-d tensor, got {factor}.")
            self.factor = factor

    def forward(self, input: Tensor) -> Tensor:
        return input * self.factor
