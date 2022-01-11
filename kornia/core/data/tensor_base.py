from typing import List, Union

from torch import Tensor


class TensorBase(Tensor):

    @staticmethod
    def __new__(cls, tensor: Tensor, *args, **kwargs):
        return super().__new__(cls, tensor, *args, **kwargs)

    def __init__(
        self, tensor: Tensor
    ) -> None:
        pass

    def transform(self, mat: Tensor):
        """Apply a transformation matrix."""
        raise NotImplementedError

    def from_tensor(self, tensor: Union[Tensor, List[Tensor]], validate: bool = True):
        raise NotImplementedError

    def to_tensor(self, tensor: Union[Tensor, List[Tensor]]):
        raise NotImplementedError
