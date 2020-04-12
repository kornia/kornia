from typing import Union
from enum import Enum


class Resample(Enum):
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2

    @classmethod
    def get(cls, value: Union[str, int, Resample]):  # type: ignore
        # TODO: Cannot define return type as Resample due to py3.6, 3.7 differences
        if type(value) == str:
            return cls[value]  # type: ignore
        if type(value) == int:
            return cls(value)
        if type(value) == cls:
            return value  # type: ignore
        raise TypeError()
