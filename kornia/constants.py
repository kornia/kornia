from typing import Union, TypeVar
from enum import Enum

import torch

__all__ = ['pi', 'Resample', 'BorderType', 'SamplePadding']

pi = torch.tensor(3.14159265358979323846)
T = TypeVar('T', bound='ConstantBase')


class ConstantBase:
    @classmethod
    def get(cls, value: Union[str, int, T]) -> T:  # type: ignore
        if type(value) == str:
            return cls[value.upper()]  # type: ignore
        if type(value) == int:
            return cls(value)  # type: ignore
        if type(value) == cls:
            return value  # type: ignore
        raise TypeError()


class Resample(ConstantBase, Enum):
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2


class BorderType(ConstantBase, Enum):
    CONSTANT = 0
    REFLECT = 1
    REPLICATE = 2
    CIRCULAR = 3


class SamplePadding(ConstantBase, Enum):
    ZEROS = 0
    BORDER = 1
    REFLECTION = 2
