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

"""The errors that kornia's input checks raise, each also the built-in exception that matches its failure.

``except BaseError`` catches every one of them. Each subclass also derives from a built-in exception, so the matching
``except`` clause catches it together with the built-in errors that other functions raise. In a call compiled with
``torch.jit.script``, a failed check raises ``torch.jit.Error`` instead, which none of these clauses catches.

Which class to use, by the failure to catch:

- Any failure of a ``KORNIA_CHECK*`` helper: :exc:`BaseError`.
- A shape or a size: :exc:`ShapeError`, a ``ValueError``; an augmentation input with an unsupported number of
  dimensions: :exc:`InputRankError`, a :exc:`ShapeError` that is also a ``RuntimeError`` and an ``AssertionError``.
- A type or a dtype: :exc:`TypeCheckError`, a ``TypeError``.
- A value or a range: :exc:`ValueCheckError`, a ``ValueError``; the host failure of a value check that an accelerator
  runs asynchronously: :exc:`AsyncCheckError`, a :exc:`ValueCheckError` that is also a ``RuntimeError``.
- Tensors on different devices: :exc:`DeviceError`, a ``ValueError``.
- A tensor without the channel layout of an image: :exc:`ImageError`, a ``ValueError``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

__all__ = [
    "AsyncCheckError",
    "BaseError",
    "DeviceError",
    "ImageError",
    "InputRankError",
    "ShapeError",
    "TypeCheckError",
    "ValueCheckError",
]


class BaseError(Exception):
    """Catch every failure of the KORNIA_CHECK helpers with this base class, which KORNIA_CHECK raises.

    ``except BaseError`` catches it and its subclasses, which the checks and some functions raise. It derives only from
    :exc:`Exception`, so neither ``except ValueError`` nor ``except TypeError`` catches it.
    """


class ShapeError(BaseError, ValueError):
    """Catch a failed shape or size check; also a ValueError, carrying both shapes.

    :func:`~kornia.core.check.KORNIA_CHECK_SHAPE`, :func:`~kornia.core.check.KORNIA_CHECK_SIZES`,
    :func:`~kornia.core.check.KORNIA_CHECK_SAME_SIZE`, :func:`~kornia.core.check.KORNIA_CHECK_SAME_SHAPE`,
    :func:`~kornia.core.check.KORNIA_CHECK_DM_DESC` and :func:`~kornia.feature.KORNIA_CHECK_LAF` raise it, and so do
    functions that check shapes themselves.

    Attributes:
        actual_shape: the shape of the tensor that failed validation, or ``None`` when the raiser gives none.
        expected_shape: the expected pattern, sizes or shape, or ``None`` when the raiser gives none.

    See Also:
        - :exc:`InputRankError`: the subclass for an augmentation input with an unsupported number of dimensions.
        - :exc:`ImageError`: a tensor without the channel layout of an image, with no shape attributes.
    """

    def __init__(
        self,
        message: str,
        *,
        actual_shape: Optional[tuple[int, ...] | list[int]] = None,
        expected_shape: Optional[list[str] | tuple[int, ...]] = None,
    ):
        super().__init__(message)
        self.actual_shape = actual_shape
        self.expected_shape = expected_shape


class InputRankError(ShapeError, RuntimeError, AssertionError):
    """Catch an augmentation input with too few or too many dimensions.

    The augmentation classes and their containers used to raise :exc:`ValueError`, :exc:`RuntimeError` or
    :exc:`AssertionError` for such an input, depending on the entry point (`#4424
    <https://github.com/kornia/kornia/issues/4424>`_), so this :exc:`ShapeError`, a :exc:`ValueError`, also derives from
    :exc:`RuntimeError` and :exc:`AssertionError`, and each of those ``except`` clauses still catches it.

    See Also:
        - :exc:`ShapeError`: the parent class, for every other shape or size failure.
    """


class TypeCheckError(BaseError, TypeError):
    """Catch a failed type or dtype check; also a TypeError, carrying both types.

    :func:`~kornia.core.check.KORNIA_CHECK_TYPE`, :func:`~kornia.core.check.KORNIA_CHECK_IS_TENSOR` and
    :func:`~kornia.core.check.KORNIA_CHECK_IS_LIST_OF_TENSOR` raise it, as does
    :func:`~kornia.core.check.KORNIA_CHECK_SAME_DEVICES` for an argument that is not a non-empty list of tensors, and so
    do functions that reject a dtype, such as an integer image.

    Attributes:
        actual_type: the actual type (or tensor dtype) that failed validation, or ``None`` when the raiser gives none.
        expected_type: the expected type (or tensor dtype), or ``None`` when the raiser gives none.

    See Also:
        - :exc:`ValueCheckError`: a value of the right type outside the accepted range.
    """

    def __init__(
        self,
        message: str,
        *,
        actual_type: Optional[type | torch.dtype] = None,
        expected_type: Optional[type | tuple[type, ...] | torch.dtype] = None,
    ):
        super().__init__(message)
        self.actual_type = actual_type
        self.expected_type = expected_type


class ValueCheckError(BaseError, ValueError):
    """Catch a failed value or range check; also a ValueError, carrying the value and range.

    :func:`~kornia.core.check.KORNIA_CHECK_IS_IMAGE` raises it for values out of its range, with the ``(min, max)`` of
    the image as ``actual_value``, and functions that check values themselves raise it as well.

    Attributes:
        actual_value: the actual value that failed validation, or ``None`` when the raiser gives none.
        expected_range: the accepted ``(min, max)``, or ``None`` when the raiser gives none.

    See Also:
        - :exc:`AsyncCheckError`: the subclass for the host failure of a value check that accelerators run
          asynchronously.
        - :exc:`TypeCheckError`: a value of the wrong type or dtype.
    """

    def __init__(
        self,
        message: str,
        *,
        actual_value: Optional[Any] = None,
        expected_range: Optional[tuple[Any, Any]] = None,
    ):
        super().__init__(message)
        self.actual_value = actual_value
        self.expected_range = expected_range


class AsyncCheckError(ValueCheckError, RuntimeError):
    """Catch the host failure of a value check that an accelerator runs as torch._assert_async.

    On CUDA, and on MPS from torch 2.13, the same check is not read back on the host: torch raises a :exc:`RuntimeError`
    by the next synchronization instead, and on CUDA that is a device-side assert after which the context cannot be
    used. This class is also a :exc:`RuntimeError`, so ``except RuntimeError`` catches a failure on every device, and a
    :exc:`ValueCheckError` for the callers that catch the host failure as a value check.

    See Also:
        - :exc:`ValueCheckError`: the parent class, for value checks that always run on the host.
        - :func:`~kornia.core.check.disable_checks`: skips these checks too, except those that keep an index in bounds.
    """


class DeviceError(BaseError, ValueError):
    """Catch a device mismatch between tensors; also a ValueError, carrying the devices.

    :func:`~kornia.core.check.KORNIA_CHECK_SAME_DEVICE` and :func:`~kornia.core.check.KORNIA_CHECK_SAME_DEVICES` raise
    it, and so do functions that check devices themselves.

    Attributes:
        actual_devices: the devices of the tensors that failed validation.
        expected_device: the expected device.

    See Also:
        - :exc:`TypeCheckError`: what :func:`~kornia.core.check.KORNIA_CHECK_SAME_DEVICES` raises for an argument that
          is not a non-empty list of tensors.
    """

    def __init__(
        self,
        message: str,
        *,
        actual_devices: Optional[list] = None,
        expected_device: Optional[Any] = None,
    ):
        super().__init__(message)
        self.actual_devices = actual_devices
        self.expected_device = expected_device


class ImageError(BaseError, ValueError):
    """Catch a tensor without the channel layout of an image; also a ValueError.

    :func:`~kornia.core.check.KORNIA_CHECK_IS_COLOR`, :func:`~kornia.core.check.KORNIA_CHECK_IS_GRAY`,
    :func:`~kornia.core.check.KORNIA_CHECK_IS_COLOR_OR_GRAY` and :func:`~kornia.core.check.KORNIA_CHECK_IS_IMAGE` raise
    it when the shape does not match the channel layout they check. It carries no attributes.

    See Also:
        - :exc:`ShapeError`: every other shape failure, with the shapes as attributes.
        - :exc:`ValueCheckError`: what :func:`~kornia.core.check.KORNIA_CHECK_IS_IMAGE` raises for values out of range.
    """
