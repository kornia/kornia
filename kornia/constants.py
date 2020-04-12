from typing import Union, TypeVar
from enum import Enum

T = TypeVar('T', bound='Resample')


class Resample(Enum):
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2

    @classmethod
    def get(cls, value: Union[str, int, T]) -> T:  # type: ignore
        if type(value) == str:
            return cls[value]  # type: ignore
        if type(value) == int:
            return cls(value)  # type: ignore
        if type(value) == cls:
            return value  # type: ignore
        raise TypeError()
