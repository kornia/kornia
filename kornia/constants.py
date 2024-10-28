import logging
from enum import Enum, EnumMeta
from typing import Iterator, Type, TypeVar, Union

import torch

from kornia.core import Tensor

__all__ = ["pi", "DType", "Resample", "BorderType", "SamplePadding", "TKEnum"]

# NOTE: to remove later
logging.basicConfig(level=logging.INFO)

pi = torch.tensor(3.14159265358979323846)


T = TypeVar("T", bound=Enum)
TKEnum = Union[str, int, T]


class _KORNIA_EnumMeta(EnumMeta):
    def __iter__(self) -> Iterator[Enum]:  # type: ignore[override]
        return super().__iter__()

    def __contains__(self, other: TKEnum[Enum]) -> bool:  # type: ignore[override]
        if isinstance(other, str):
            return any(val.name.upper() == other.upper() for val in self)

        elif isinstance(other, int):
            return any(val.value == other for val in self)

        return any(val == other for val in self)

    def __repr__(self) -> str:
        return " | ".join(f"{self.__name__}.{val.name}" for val in self)


def _get(cls: Type[T], value: TKEnum[T]) -> T:
    if isinstance(value, str):
        return cls[value.upper()]

    elif isinstance(value, int):
        return cls(value)

    elif isinstance(value, cls):
        return value

    raise TypeError(
        f"The `.get` method from `{cls}` expects a value with type `str`, `int` or `{cls}`. Gotcha {type(value)}"
    )


class Resample(Enum, metaclass=_KORNIA_EnumMeta):
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2

    @classmethod
    def get(cls, value: TKEnum["Resample"]) -> "Resample":
        return _get(cls, value)


class BorderType(Enum, metaclass=_KORNIA_EnumMeta):
    CONSTANT = 0
    REFLECT = 1
    REPLICATE = 2
    CIRCULAR = 3

    @classmethod
    def get(cls, value: TKEnum["BorderType"]) -> "BorderType":
        return _get(cls, value)


class SamplePadding(Enum, metaclass=_KORNIA_EnumMeta):
    ZEROS = 0
    BORDER = 1
    REFLECTION = 2

    @classmethod
    def get(cls, value: TKEnum["SamplePadding"]) -> "SamplePadding":
        return _get(cls, value)


class DType(Enum, metaclass=_KORNIA_EnumMeta):
    INT64 = 0
    FLOAT16 = 1
    FLOAT32 = 2
    FLOAT64 = 3

    @classmethod
    def get(cls, value: Union[str, int, torch.dtype, Tensor, "DType"]) -> "DType":
        if isinstance(value, torch.dtype):
            return cls[str(value).upper()[6:]]

        elif isinstance(value, Tensor):
            return cls(int(value.item()))

        elif isinstance(value, str):
            return cls[value.upper()]

        elif isinstance(value, int):
            return cls(value)

        elif isinstance(value, cls):
            return value

        raise TypeError(f"Invalid identifier {value} with type {type(value)}.")

    @classmethod
    def to_torch(cls, value: TKEnum["DType"]) -> torch.dtype:
        data = cls.get(value=value)

        if data == DType.INT64:
            return torch.long

        elif data == DType.FLOAT16:
            return torch.float16

        elif data == DType.FLOAT32:
            return torch.float32

        elif data == DType.FLOAT64:
            return torch.float64

        raise ValueError


# TODO: (low-priority) add INPUT3D, MASK3D, BBOX3D, LAFs etc.
class DataKey(Enum, metaclass=_KORNIA_EnumMeta):
    IMAGE = 0
    INPUT = 0
    MASK = 1
    BBOX = 2
    BBOX_XYXY = 3
    BBOX_XYWH = 4
    KEYPOINTS = 5
    LABEL = 6
    CLASS = 6

    @classmethod
    def get(cls, value: TKEnum["DataKey"]) -> "DataKey":
        return _get(cls, value)
