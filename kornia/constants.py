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

r"""Name the interpolation, padding, border, dtype and data-kind options of the augmentations, as enums or strings.

Each enum's ``get`` classmethod takes a member, its name in any case or its integer value and returns the member, so
an argument typed ``TKEnum`` (``str | int | Enum``) accepts all three. A name that names no member raises an error that
is both a ``KeyError`` and a ``ValueError``, an integer that is no member's value a ``ValueError``, and another type a
``TypeError``. ``"bilinear" in Resample`` tests a name case-insensitively, and ``1 in Resample`` a value.

Which class to use:

- The interpolation of a resampling augmentation (``resample``): :class:`Resample`.
- How a warping augmentation fills pixels that map outside the input (``padding_mode``): :class:`SamplePadding`.
- The border mode of a blur augmentation (``border_type``): :class:`BorderType`.
- The kind of each input of an augmentation container (``data_keys``): :class:`DataKey`.
- A tensor dtype as an integer code: :class:`DType`.
- The ``str | int | Enum`` type of the arguments that ``get`` reads: ``TKEnum``.
- :math:`\pi` as a 0-d tensor of the default dtype and device at import: ``pi``.
"""

from enum import Enum, EnumMeta
from typing import Iterator, Type, TypeVar, Union

import torch

__all__ = ["BorderType", "DType", "DataKey", "Resample", "SamplePadding", "TKEnum", "pi"]

pi = torch.tensor(3.14159265358979323846)


_T = TypeVar("_T", bound=Enum)
TKEnum = Union[str, int, _T]


class _KORNIA_EnumMeta(EnumMeta):
    """Custom metaclass for enums to support string and integer comparisons."""

    def __iter__(self) -> Iterator[Enum]:
        return super().__iter__()

    def __contains__(self, other: TKEnum[Enum]) -> bool:
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
    # try/except, not ``name in cls.__members__``: dynamo on torch 2.5.1 cannot trace ``in`` on the members mapping,
    # and augmentations look names up inside ``forward`` (a per-call ``border_type``, ``data_keys``).
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

    if isinstance(value, bool):
        raise TypeError(
            f"`{cls.__name__}.get` expects a `str`, `int` or `{cls.__name__}` value, not a `bool`. "
            f"Got {value!r} of type `{type(value).__name__}`."
        )

    if isinstance(value, int):
        return cls(value)

    if isinstance(value, cls):
        return value

    raise TypeError(
        f"`{cls.__name__}.get` expects a `str`, `int` or `{cls.__name__}` value. "
        f"Got {value!r} of type `{type(value).__name__}`."
    )


class Resample(Enum, metaclass=_KORNIA_EnumMeta):
    """Name the interpolation mode of a resampling augmentation: nearest, bilinear or bicubic.

    The geometric augmentations read their ``resample`` argument with ``Resample.get``, so
    :class:`~kornia.augmentation.RandomAffine` takes ``resample="nearest"`` as well as ``Resample.NEAREST``.

    See Also:
        - :class:`SamplePadding`: the ``padding_mode`` of the same warping augmentations.
        - :class:`BorderType`: the border mode of the blur augmentations.

    """

    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2

    @classmethod
    def get(cls, value: TKEnum["Resample"]) -> "Resample":
        """Return the Resample member matching a name, a value or a member."""
        return _get(cls, value)


class BorderType(Enum, metaclass=_KORNIA_EnumMeta):
    """Name the border mode of a blur augmentation: constant, reflect, replicate or circular padding.

    The blur augmentations, such as :class:`~kornia.augmentation.RandomGaussianBlur`,
    :class:`~kornia.augmentation.RandomBoxBlur` and :class:`~kornia.augmentation.RandomMotionBlur`, read their
    ``border_type`` argument with ``BorderType.get`` and pass the lower-case name to their filter.

    See Also:
        - :class:`SamplePadding`: how a warping augmentation fills pixels that map outside the input.
        - :class:`Resample`: the interpolation mode of the resampling augmentations.

    """

    CONSTANT = 0
    REFLECT = 1
    REPLICATE = 2
    CIRCULAR = 3

    @classmethod
    def get(cls, value: TKEnum["BorderType"]) -> "BorderType":
        """Return the BorderType member matching a name, a value or a member."""
        return _get(cls, value)


class SamplePadding(Enum, metaclass=_KORNIA_EnumMeta):
    """Name how a warping augmentation fills pixels outside the input: zeros, border, reflection or a fill value.

    :class:`~kornia.augmentation.RandomAffine`, :class:`~kornia.augmentation.RandomThinPlateSpline` and the shear and
    translation augmentations read their ``padding_mode`` argument with ``SamplePadding.get``. ``ZEROS``, ``BORDER`` and
    ``REFLECTION`` are the padding modes of :func:`torch.nn.functional.grid_sample`; ``FILL`` fills with the
    ``fill_value`` of an augmentation that takes one, such as :class:`~kornia.augmentation.RandomAffine`.

    See Also:
        - :class:`Resample`: the interpolation mode of the same warping augmentations.
        - :class:`BorderType`: the border mode of the blur augmentations.

    """

    ZEROS = 0
    BORDER = 1
    REFLECTION = 2
    FILL = 3

    @classmethod
    def get(cls, value: TKEnum["SamplePadding"]) -> "SamplePadding":
        """Return the SamplePadding member matching a name, a value or a member."""
        return _get(cls, value)


class DType(Enum, metaclass=_KORNIA_EnumMeta):
    """Name a tensor dtype by an integer code, so that it can be stored in a tensor of augmentation parameters.

    ``DType.get`` also takes a ``torch.dtype`` or a one-element tensor holding a value, and ``DType.to_torch(value)``
    returns the matching ``torch.dtype``; a torch dtype without a member, such as ``torch.uint8``, raises. The mix
    augmentations, such as :class:`~kornia.augmentation.RandomMixUpV2`, record the input dtype in their ``dtype``
    parameter this way.

    """

    INT64 = 0
    FLOAT16 = 1
    FLOAT32 = 2
    FLOAT64 = 3
    BFLOAT16 = 4

    @classmethod
    def get(cls, value: Union[str, int, torch.dtype, torch.Tensor, "DType"]) -> "DType":
        """Return the DType member matching a name, a value, a torch dtype, a tensor or a member."""
        if isinstance(value, torch.dtype):
            return _get_by_name(cls, str(value)[6:], value)

        if isinstance(value, bool):
            raise TypeError(
                "`DType.get` expects a `str`, `int`, `torch.dtype`, `torch.Tensor` or `DType` value, not a `bool`. "
                f"Got {value!r} of type `{type(value).__name__}`."
            )

        if isinstance(value, torch.Tensor):
            item = value.item()
            if not float(item).is_integer():
                raise TypeError(f"`DType.get` expects a tensor with an integral scalar value. Got {value!r}.")
            return cls(int(item))

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
        """Return the torch dtype matching a DType name, value or member."""
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
    """Name the kind of each input of an augmentation container: image, mask, boxes, keypoints or labels.

    ``AugmentationSequential(..., data_keys=[DataKey.INPUT, "mask"])`` takes one member, or its name in any case, per
    input. ``INPUT`` is an alias of ``IMAGE`` and ``CLASS`` of ``LABEL``. ``BBOX`` holds ``(B, N, 4, 2)`` box corners,
    ``BBOX_XYXY`` and ``BBOX_XYWH`` ``(B, N, 4)`` boxes, and ``KEYPOINTS`` ``(B, N, 2)`` points.

    See Also:
        - :class:`~kornia.augmentation.container.AugmentationSequential`: the container whose ``data_keys`` these name.

    """

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
        """Return the DataKey member matching a name, a value or a member."""
        return _get(cls, value)
