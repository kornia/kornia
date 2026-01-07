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

__all__ = [
    "BaseError",
    "DeviceError",
    "ImageError",
    "ShapeError",
    "TypeCheckError",
    "ValueCheckError",
]


class BaseError(Exception):
    """Base exception class for all Kornia errors."""

    pass


class ShapeError(BaseError):
    """Raised when tensor shape validation fails.

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


class TypeCheckError(BaseError):
    """Raised when type validation fails.

    Attributes:
        actual_type: The actual type that failed validation.
        expected_type: The expected type.
    """

    def __init__(
        self,
        message: str,
        *,
        actual_type: Optional[type] = None,
        expected_type: Optional[type | tuple[type, ...]] = None,
    ):
        super().__init__(message)
        self.actual_type = actual_type
        self.expected_type = expected_type


class ValueCheckError(BaseError):
    """Raised when value/range validation fails.

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


class DeviceError(BaseError):
    """Raised when device mismatch validation fails.

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


class ImageError(BaseError):
    """Raised when image-specific validation fails."""

    pass
