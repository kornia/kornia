from typing import Union, TypeVar
from enum import Enum, EnumMeta

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


class EnumMetaFlags(EnumMeta):
    def __contains__(self, value: Union[str, int, T]) -> bool:
        try:
            if type(value) == str:
                self[value.upper()]  # type: ignore
                return True
            if type(value) == int:
                self(value)  # type: ignore
                return True
            if type(value) == self:
                return True
        except TypeError as e:
            return False
        return False

    def __repr__(self):
        return ' | '.join(f"{self.__name__}.{val.name}" for val in self)


class Resample(ConstantBase, Enum, metaclass=EnumMetaFlags):
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2


class BorderType(ConstantBase, Enum, metaclass=EnumMetaFlags):
    CONSTANT = 0
    REFLECT = 1
    REPLICATE = 2
    CIRCULAR = 3


class SamplePadding(ConstantBase, Enum, metaclass=EnumMetaFlags):
    ZEROS = 0
    BORDER = 1
    REFLECTION = 2


class InputType(ConstantBase, Enum, metaclass=EnumMetaFlags):
    INPUT = 0
    MASK = 1
    BBOX = 2
    BBOX_XYXY = 3
    BBOX_XYHW = 4
    KEYPOINTS = 5
