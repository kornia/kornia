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

"""Custom exception classes for Kornia validation checks."""

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
    """Base class of the exceptions raised by kornia's ``KORNIA_CHECK*`` validation helpers and a few functions.

    A failed ``KORNIA_CHECK`` raises it directly; it is neither a :exc:`ValueError` nor a :exc:`TypeError`. Each
    subclass also derives from the one of those two built-ins that matches its failure, so the matching ``except``
    clause catches the subclass as well. :exc:`AsyncCheckError` is also a :exc:`RuntimeError`.
    """


class ShapeError(BaseError, ValueError):
    """Raised when tensor shape validation fails. Also a :exc:`ValueError`.

    Attributes:
        actual_shape: The actual shape of the tensor that failed validation.
        expected_shape: The expected shape specification.
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
    """Raised when an input has a number of dimensions the operation does not accept. Also a :exc:`ValueError`.

    The augmentation classes and their containers raise it for one user error: an image or a volume with too few or
    too many dimensions. They used to raise :exc:`ValueError`, :exc:`RuntimeError` or :exc:`AssertionError` for it,
    depending on the entry point (`#4424 <https://github.com/kornia/kornia/issues/4424>`_), so this class also
    derives from :exc:`RuntimeError` and :exc:`AssertionError`, and each of those ``except`` clauses still catches it.
    """


class TypeCheckError(BaseError, TypeError):
    """Raised when type validation fails. Also a :exc:`TypeError`.

    Attributes:
        actual_type: The actual type (or tensor dtype) that failed validation.
        expected_type: The expected type (or tensor dtype).
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
    """Raised when value/range validation fails. Also a :exc:`ValueError`.

    Attributes:
        actual_value: The actual value that failed validation.
        expected_range: The expected value range (min, max).
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
    """Raised on the host by a value check that an accelerator runs as ``torch._assert_async``.

    On CUDA, and on MPS from torch 2.13, the same check is not read back on the host: torch raises a
    :exc:`RuntimeError` by the next synchronization instead, and on CUDA that is a device-side assert after which the
    context cannot be used. This class is also a :exc:`RuntimeError`, so ``except RuntimeError`` handles a failure on
    every device, and a :exc:`ValueCheckError` for the callers that catch the host failure as a value check.
    """


class DeviceError(BaseError, ValueError):
    """Raised when device mismatch validation fails. Also a :exc:`ValueError`.

    Attributes:
        actual_devices: The actual device(s) that failed validation.
        expected_device: The expected device.
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
    """Raised when image-specific validation fails. Also a :exc:`ValueError`."""
