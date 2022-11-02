from enum import Enum, EnumMeta
from typing import Any, Iterator, Union

import torch

from kornia.core import Tensor

__all__ = ['pi', 'DType', 'Resample', 'BorderType', 'SamplePadding']

pi = torch.tensor(3.14159265358979323846)


class EnumMetaFlags(EnumMeta):
    def __iter__(self) -> Iterator[Enum]:  # type: ignore[override]
        return super().__iter__()

    def __contains__(self, other: Union[str, int, Enum]) -> bool:  # type: ignore[override]
        if isinstance(other, str):
            return any(val.name == other.upper() for val in self)
        elif isinstance(other, int):
            return any(val.value == other for val in self)

        return any(val == other for val in self)

    def __repr__(self):
        return ' | '.join(f"{self.__name__}.{val.name}" for val in self)


class ConstantBase(Enum, metaclass=EnumMetaFlags):
    @classmethod
    def get(cls, value: Union[str, int, Any]) -> Any:
        if isinstance(value, str):
            return cls[value.upper()]
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, cls):
            return value
        raise TypeError()


class Resample(ConstantBase):
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2


class BorderType(ConstantBase):
    CONSTANT = 0
    REFLECT = 1
    REPLICATE = 2
    CIRCULAR = 3


class SamplePadding(ConstantBase):
    ZEROS = 0
    BORDER = 1
    REFLECTION = 2


class DType(ConstantBase):
    INT64 = 0
    FLOAT16 = 1
    FLOAT32 = 2
    FLOAT64 = 3

    @classmethod
    def get(cls, value: Union[str, int, torch.dtype, Tensor, 'DType']) -> 'DType':  # type: ignore[override]
        if isinstance(value, torch.dtype):
            value = str(value).upper()  # Convert to str
        if isinstance(value, Tensor):
            value = int(value.item())  # Convert to int

        if isinstance(value, str):
            if value.upper().startswith("TORCH."):
                return cls[value.upper()[6:]]
            return cls[value.upper()]

        if isinstance(value, int):
            return cls(value)
        if isinstance(value, cls):
            return value
        raise TypeError(f"Invalid identifier {value}.")

    @classmethod
    def to_torch(cls, value: Union[str, int, 'DType']) -> torch.dtype:
        data = cls.get(value=value)
        if data == DType.INT64:
            return torch.long
        if data == DType.FLOAT16:
            return torch.float16
        if data == DType.FLOAT32:
            return torch.float32
        if data == DType.FLOAT64:
            return torch.float64
        raise ValueError()


# TODO: (low-priority) add INPUT3D, MASK3D, BBOX3D, LAFs etc.
class DataKey(ConstantBase):
    INPUT = 0
    MASK = 1
    BBOX = 2
    BBOX_XYXY = 3
    BBOX_XYWH = 4
    KEYPOINTS = 5
    CLASS = 6
