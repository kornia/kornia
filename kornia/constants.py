# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from enum import Enum, EnumMeta
from typing import Iterator, Type, TypeVar, Union

import torch

__all__ = ["BorderType", "DType", "Resample", "SamplePadding", "TKEnum", "pi"]

pi = torch.tensor(3.14159265358979323846)


_T = TypeVar("_T", bound=Enum)
TKEnum = Union[str, int, _T]


class _KORNIA_EnumMeta(EnumMeta):
    """Custom metaclass for enums to support string and integer comparisons."""

    def __iter__(self) -> Iterator[Enum]:  # type: ignore[override]
        return super().__iter__()

    def __contains__(self, other: TKEnum[Enum]) -> bool:  # type: ignore[override]
        if isinstance(other, str):
            return any(val.name.upper() == other.upper() for val in self)

        if isinstance(other, int):
            return any(val.value == other for val in self)

        return any(val == other for val in self)

    def __repr__(self) -> str:
        return " | ".join(f"{self.__name__}.{val.name}" for val in self)


class _UnknownNameError(KeyError, ValueError):
    """Raised by the ``.get`` of a kornia enum for a name that names no member.

    It is a :exc:`ValueError`, like the error for an integer that is no member's value, and a :exc:`KeyError`, like a
    failed ``Enum[name]`` lookup, so either ``except`` clause catches it.
    """

    def __str__(self) -> str:
        # `KeyError.__str__` would repr-quote the whole message
        return Exception.__str__(self)


def _get_by_name(cls: Type[_T], name: str, value: object) -> _T:
    try:
        return cls[name.upper()]
    except KeyError:
        valid = ", ".join(repr(member_name.lower()) for member_name in cls.__members__)
        raise _UnknownNameError(
            f"{value!r} is not a valid {cls.__name__}. Expected one of {valid} (case-insensitive)."
        ) from None


def _get(cls: Type[_T], value: TKEnum[_T]) -> _T:
    if isinstance(value, str):
        return _get_by_name(cls, value, value)

    if isinstance(value, int):
        return cls(value)

    if isinstance(value, cls):
        return value

    raise TypeError(
        f"`{cls.__name__}.get` expects a `str`, `int` or `{cls.__name__}` value. "
        f"Got {value!r} of type `{type(value).__name__}`."
    )


class Resample(Enum, metaclass=_KORNIA_EnumMeta):
    """Represent the resampling mode for image interpolation."""

    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2

    @classmethod
    def get(cls, value: TKEnum["Resample"]) -> "Resample":
        return _get(cls, value)


class BorderType(Enum, metaclass=_KORNIA_EnumMeta):
    """Represent the border padding mode for image operations."""

    CONSTANT = 0
    REFLECT = 1
    REPLICATE = 2
    CIRCULAR = 3

    @classmethod
    def get(cls, value: TKEnum["BorderType"]) -> "BorderType":
        return _get(cls, value)


class SamplePadding(Enum, metaclass=_KORNIA_EnumMeta):
    """Represent the padding mode used during spatial sampling."""

    ZEROS = 0
    BORDER = 1
    REFLECTION = 2
    FILL = 3

    @classmethod
    def get(cls, value: TKEnum["SamplePadding"]) -> "SamplePadding":
        return _get(cls, value)


class DType(Enum, metaclass=_KORNIA_EnumMeta):
    """Represent the internal data types used across the library."""

    INT64 = 0
    FLOAT16 = 1
    FLOAT32 = 2
    FLOAT64 = 3
    BFLOAT16 = 4

    @classmethod
    def get(cls, value: Union[str, int, torch.dtype, torch.Tensor, "DType"]) -> "DType":
        if isinstance(value, torch.dtype):
            return _get_by_name(cls, str(value)[6:], value)

        if isinstance(value, torch.Tensor):
            return cls(int(value.item()))

        if isinstance(value, str):
            return _get_by_name(cls, value, value)

        if isinstance(value, int):
            return cls(value)

        if isinstance(value, cls):
            return value

        raise TypeError(
            "`DType.get` expects a `str`, `int`, `torch.dtype`, `torch.Tensor` or `DType` value. "
            f"Got {value!r} of type `{type(value).__name__}`."
        )

    @classmethod
    def to_torch(cls, value: TKEnum["DType"]) -> torch.dtype:
        data = cls.get(value=value)

        if data == DType.INT64:
            return torch.long

        if data == DType.FLOAT16:
            return torch.float16

        if data == DType.FLOAT32:
            return torch.float32

        if data == DType.FLOAT64:
            return torch.float64

        if data == DType.BFLOAT16:
            return torch.bfloat16

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
