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

"""The testing package contains testing-specific utilities."""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, TypeVar, cast

import torch
from torch import float16, float32, float64
from typing_extensions import TypeGuard

from kornia.core.exceptions import (
    BaseError,
    DeviceError,
    ImageError,
    ShapeError,
    TypeCheckError,
    ValueCheckError,
)

__all__ = [
    "KORNIA_CHECK",
    "KORNIA_CHECK_DM_DESC",
    "KORNIA_CHECK_IS_COLOR",
    "KORNIA_CHECK_IS_GRAY",
    "KORNIA_CHECK_IS_IMAGE",
    "KORNIA_CHECK_IS_LIST_OF_TENSOR",
    "KORNIA_CHECK_IS_TENSOR",
    "KORNIA_CHECK_LAF",
    "KORNIA_CHECK_SAME_DEVICE",
    "KORNIA_CHECK_SAME_DEVICES",
    "KORNIA_CHECK_SHAPE",
    "KORNIA_CHECK_TYPE",
    "KORNIA_UNWRAP",
    "BaseError",
    "DeviceError",
    "ImageError",
    "ShapeError",
    "TypeCheckError",
    "ValueCheckError",
    "are_checks_enabled",
    "disable_checks",
    "enable_checks",
]


def _should_enable_checks() -> bool:
    """Determine if checks should be enabled.

    Returns:
        True if checks should be enabled, False otherwise.

    Checks are enabled by default in debug mode (normal Python execution).
    Checks are disabled when:
    - Running with `python -O` (optimized mode)
    - Environment variable KORNIA_CHECKS=0 is set
    """
    env_var = os.getenv("KORNIA_CHECKS", None)
    if env_var is not None:
        # Explicit override via environment variable
        return env_var.lower() in ("1", "true", "yes", "on")
    # Default: enabled in debug mode, disabled in optimized mode
    return __debug__


# Module-level flag - evaluated once at import time, but can be changed at runtime
_KORNIA_CHECKS_ENABLED: bool = _should_enable_checks()


def are_checks_enabled() -> bool:
    """Check if validation is currently enabled.

    Returns:
        True if checks are enabled, False otherwise.

    Example:
        >>> are_checks_enabled()
        True
    """
    return _KORNIA_CHECKS_ENABLED


def disable_checks() -> None:
    """Disable all Kornia validation checks for production.

    Note:
<<<<<<< HEAD
        This function can override the initial setting determined at import time.
        The module-level flag is evaluated once at import time for performance, but
        can be changed at runtime via this function or `enable_checks()`.
=======
        This function has no effect if checks were disabled at import time
        (via `python -O` or KORNIA_CHECKS=0). The module-level flag is
        evaluated once at import time for performance.
>>>>>>> 846b6f22 (implement custom errors)

    Example:
        >>> disable_checks()
        >>> are_checks_enabled()
        False
    """
    global _KORNIA_CHECKS_ENABLED  # noqa: PLW0603
    _KORNIA_CHECKS_ENABLED = False


def enable_checks() -> None:
    """Enable all Kornia validation checks.

<<<<<<< HEAD
    Note:
        This function can override the initial setting determined at import time.
        The module-level flag is evaluated once at import time for performance, but
        can be changed at runtime via this function or `disable_checks()`.

=======
>>>>>>> 846b6f22 (implement custom errors)
    Example:
        >>> enable_checks()
        >>> are_checks_enabled()
        True
    """
    global _KORNIA_CHECKS_ENABLED  # noqa: PLW0603
    _KORNIA_CHECKS_ENABLED = True


# Logger api


def KORNIA_CHECK_SHAPE(x: torch.Tensor, shape: list[str], msg: Optional[str] = None, raises: bool = True) -> bool:
    """Check whether a tensor has a specified shape.

    The shape can be specified with a implicit or explicit list of strings.
    The guard also check whether the variable is a type `Tensor`.

    Args:
        x: the tensor to evaluate.
        shape: a list with strings with the expected shape.
        msg: optional custom message to append to error.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        ShapeError: if the input tensor does not have the expected shape and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["B", "C", "H", "W"])  # implicit
        True

        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["2", "3", "H", "W"])  # explicit
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if "*" == shape[0]:
        shape_to_check = shape[1:]
        x_shape_to_check = x.shape[-len(shape) + 1 :]
    elif "*" == shape[-1]:
        shape_to_check = shape[:-1]
        x_shape_to_check = x.shape[: len(shape) - 1]
    else:
        shape_to_check = shape
        x_shape_to_check = x.shape

    if len(x_shape_to_check) != len(shape_to_check):
        if raises:
            expected_dims = len(shape_to_check)
            actual_dims = len(x_shape_to_check)
            error_msg = f"Shape dimension mismatch: expected {expected_dims} dimensions, got {actual_dims}.\n"
            error_msg += f"  Expected shape: {shape}\n"
            x_shape_list = list(x.shape)
            error_msg += f"  Actual shape: {x_shape_list}"
            if msg is not None:
                error_msg += f"\n  {msg}"
            raise ShapeError(
                error_msg,
                actual_shape=x_shape_list,
                expected_shape=shape,
            )
        else:
            return False

    for i in range(len(x_shape_to_check)):
        # The voodoo below is because torchscript does not like
        # that dim can be both int and str
        dim_: str = shape_to_check[i]
        if not dim_.isnumeric():
            continue
        dim = int(dim_)
        if x_shape_to_check[i] != dim:
            if raises:
                error_msg = f"Shape mismatch at dimension {i}: expected {dim}, got {x_shape_to_check[i]}.\n"
                error_msg += f"  Expected shape: {shape}\n"
                x_shape_list = list(x.shape)
                error_msg += f"  Actual shape: {x_shape_list}"
                if msg is not None:
                    error_msg += f"\n  {msg}"
                raise ShapeError(
                    error_msg,
                    actual_shape=x_shape_list,
                    expected_shape=shape,
                )
            else:
                return False
    return True


def KORNIA_CHECK(condition: bool, msg: Optional[str] = None, raises: bool = True) -> bool:
    """Check any arbitrary boolean condition.

    Args:
        condition: the condition to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        BaseError: if the condition is not met and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK(x.shape[-2:] == (3, 3), "Invalid homography")
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if not condition:
        if raises:
            if msg is None:
                error_msg = "Validation condition failed"
            else:
                error_msg = msg
            raise BaseError(error_msg)
        return False
    return True


def KORNIA_UNWRAP(maybe_obj: object, typ: Any) -> Any:
    """Unwrap an optional contained value that may or not be present.

    Args:
        maybe_obj: the object to unwrap.
        typ: expected type after unwrap.

    """
    # TODO: this function will change after kornia/pr#1987
    return cast(typ, maybe_obj)


T = TypeVar("T", bound=type)


def KORNIA_CHECK_TYPE(
    x: object, typ: T | tuple[T, ...], msg: Optional[str] = None, raises: bool = True
) -> TypeGuard[T]:
    """Check the type of an aribratry variable.

    Args:
        x: any input variable.
        typ: the expected type of the variable.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeCheckError: if the input variable does not match with the expected and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> KORNIA_CHECK_TYPE("foo", str, "Invalid string")
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if not isinstance(x, typ):
        if raises:
            expected_type_str = typ.__name__ if not isinstance(typ, tuple) else " | ".join(t.__name__ for t in typ)
            error_msg = f"Type mismatch: expected {expected_type_str}, got {type(x).__name__}."
            if msg is not None:
                error_msg += f"\n  {msg}"
            raise TypeCheckError(
                error_msg,
                actual_type=type(x),
                expected_type=typ,
            )
        return False
    return True


def KORNIA_CHECK_IS_TENSOR(x: object, msg: Optional[str] = None, raises: bool = True) -> TypeGuard[torch.Tensor]:
    """Check the input variable is a Tensor.

    Args:
        x: any input variable.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeCheckError: if the input variable does not match with the expected and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_IS_TENSOR(x, "Invalid tensor")
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if not isinstance(x, torch.Tensor):
        if raises:
            error_msg = f"Type mismatch: expected Tensor, got {type(x).__name__}."
            if msg is not None:
                error_msg += f"\n  {msg}"
            raise TypeCheckError(
                error_msg,
                actual_type=type(x),
                expected_type=torch.Tensor,
            )
        return False
    return True


def KORNIA_CHECK_IS_LIST_OF_TENSOR(x: Optional[Sequence[object]], raises: bool = True) -> TypeGuard[list[torch.Tensor]]:
    """Check the input variable is a List of Tensors.

    Args:
        x: Any sequence of objects
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeCheckError: if the input variable does not match with the expected and raises is True.

    Return:
        True if the input is a list of Tensors, otherwise return False.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_IS_LIST_OF_TENSOR(x, raises=False)
        False
        >>> KORNIA_CHECK_IS_LIST_OF_TENSOR([x])
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    are_tensors = isinstance(x, list) and all(isinstance(d, torch.Tensor) for d in x)
    if not are_tensors:
        if raises:
            error_msg = f"Type mismatch: expected list[Tensor], got {type(x).__name__}."
            raise TypeCheckError(
                error_msg,
                actual_type=type(x),
                expected_type=list[torch.Tensor],
            )
        return False
    return True


def KORNIA_CHECK_SAME_DEVICE(x: torch.Tensor, y: torch.Tensor, raises: bool = True) -> bool:
    """Check whether two tensor in the same device.

    Args:
        x: first tensor to evaluate.
        y: second tensor to evaluate.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        DeviceError: if the two tensors are not in the same device and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(1, 3, 1)
        >>> KORNIA_CHECK_SAME_DEVICE(x1, x2)
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if x.device != y.device:
        if raises:
            error_msg = "Device mismatch: tensors must be on the same device.\n"
            error_msg += f"  First tensor device: {x.device}\n"
            error_msg += f"  Second tensor device: {y.device}"
            raise DeviceError(
                error_msg,
                actual_devices=[x.device, y.device],
                expected_device=x.device,
            )
        return False
    return True


def KORNIA_CHECK_SAME_DEVICES(tensors: list[torch.Tensor], msg: Optional[str] = None, raises: bool = True) -> bool:
    """Check whether a list provided tensors live in the same device.

    Args:
        tensors: a list of tensors.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        DeviceError: if all the tensors are not in the same device and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(1, 3, 1)
        >>> KORNIA_CHECK_SAME_DEVICES([x1, x2], "Tensors not in the same device")
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    KORNIA_CHECK(isinstance(tensors, list) and len(tensors) >= 1, "Expected a list with at least one element", raises)
    if not all(tensors[0].device == x.device for x in tensors):
        if raises:
            devices = [x.device for x in tensors]
            error_msg = "Device mismatch: all tensors must be on the same device.\n"
            error_msg += f"  Expected device: {tensors[0].device}\n"
            error_msg += f"  Actual devices: {devices}"
            if msg is not None:
                error_msg += f"\n  {msg}"
            raise DeviceError(
                error_msg,
                actual_devices=devices,
                expected_device=tensors[0].device,
            )
        return False
    return True


def KORNIA_CHECK_SAME_SHAPE(x: torch.Tensor, y: torch.Tensor, raises: bool = True) -> bool:
    """Check whether two tensor have the same shape.

    Args:
        x: first tensor to evaluate.
        y: second tensor to evaluate.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        ShapeError: if the two tensors have not the same shape and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_SAME_SHAPE(x1, x2)
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if x.shape != y.shape:
        if raises:
            error_msg = "Shape mismatch: tensors must have the same shape.\n"
            x_shape_list = list(x.shape)
            y_shape_list = list(y.shape)
            error_msg += f"  First tensor shape: {x_shape_list}\n"
            error_msg += f"  Second tensor shape: {y_shape_list}"
            raise ShapeError(
                error_msg,
                actual_shape=x_shape_list,
                expected_shape=y_shape_list,
            )
        return False
    return True


def KORNIA_CHECK_IS_COLOR(x: torch.Tensor, msg: Optional[str] = None, raises: bool = True) -> bool:
    """Check whether an image tensor is a color images.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        ImageError: if the input tensor does not have a shape :math:`(3,H,W)` and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_IS_COLOR(img, "Image is not color")
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if len(x.shape) < 3 or x.shape[-3] != 3:
        if raises:
            error_msg = f"Not a color tensor. Got: {type(x)}."
            if msg is not None:
                error_msg += f"\n{msg}"
            raise ImageError(error_msg)
        return False
    return True


def KORNIA_CHECK_IS_GRAY(x: torch.Tensor, msg: Optional[str] = None, raises: bool = True) -> bool:
    """Check whether an image tensor is grayscale.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        ImageError: if the tensor does not have a shape :math:`(1,H,W)` or :math:`(H,W)` and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> img = torch.rand(2, 1, 4, 4)
        >>> KORNIA_CHECK_IS_GRAY(img, "Image is not grayscale")
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if len(x.shape) < 2 or (len(x.shape) >= 3 and x.shape[-3] != 1):
        if raises:
            error_msg = f"Not a gray tensor. Got: {type(x)}."
            if msg is not None:
                error_msg += f"\n{msg}"
            raise ImageError(error_msg)
        return False
    return True


def KORNIA_CHECK_IS_COLOR_OR_GRAY(x: torch.Tensor, msg: Optional[str] = None, raises: bool = True) -> bool:
    """Check whether an image tensor is grayscale or color.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        ImageError: if the tensor does not have a shape :math:`(1,H,W)` or :math:`(3,H,W)` and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_IS_COLOR_OR_GRAY(img, "Image is not color or grayscale")
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if len(x.shape) < 3 or x.shape[-3] not in [1, 3]:
        if raises:
            error_msg = f"Not a color or gray tensor. Got: {type(x)}."
            if msg is not None:
                error_msg += f"\n{msg}"
            raise ImageError(error_msg)
        return False
    return True


def KORNIA_CHECK_IS_IMAGE(x: torch.Tensor, msg: Optional[str] = None, raises: bool = True, bits: int = 8) -> bool:
    """Check whether an image tensor is ranged properly [0, 1] for float or [0, 2 ** bits] for int.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.
        bits: the image bits. The default checks if given integer input image is an
            8-bit image (0-255) or not.

    Raises:
        ImageError: if the tensor shape is invalid and raises is True.
        ValueCheckError: if the tensor value range is invalid and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_IS_IMAGE(img, "It is not an image")
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    # Combine the color or gray check with the range check
    if not raises and not KORNIA_CHECK_IS_COLOR_OR_GRAY(x, msg, raises):
        return False

    amin, amax = torch.aminmax(x)

    if x.dtype in (float16, float32, float64):
        invalid = (amin < 0) | (amax > 1)
    else:
        max_int_value = (1 << bits) - 1
        invalid = (amin < 0) | (amax > max_int_value)

    if invalid.item():
        return _handle_invalid_range(msg, raises, amin, amax)

    return True


def KORNIA_CHECK_DM_DESC(desc1: torch.Tensor, desc2: torch.Tensor, dm: torch.Tensor, raises: bool = True) -> bool:
    """Check whether the provided descriptors match with a distance matrix.

    Args:
        desc1: first descriptor tensor to evaluate.
        desc2: second descriptor tensor to evaluate.
        dm: distance matrix tensor to evaluate.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        ShapeError: if the descriptors shape do not match with the distance matrix and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> desc1 = torch.rand(4)
        >>> desc2 = torch.rand(8)
        >>> dm = torch.rand(4, 8)
        >>> KORNIA_CHECK_DM_DESC(desc1, desc2, dm)
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
        if raises:
            expected_shape = (desc1.size(0), desc2.size(0))
            desc1_size = desc1.size(0)
            desc2_size = desc2.size(0)
            error_msg = "Distance matrix shape mismatch.\n"
            error_msg += (
                f"  Expected shape: {expected_shape} (from desc1.shape[0]={desc1_size}, desc2.shape[0]={desc2_size})\n"
            )
            dm_shape_list = list(dm.shape)
            desc1_shape_list = list(desc1.shape)
            desc2_shape_list = list(desc2.shape)
            error_msg += f"  Actual shape: {dm_shape_list}\n"
            error_msg += f"  desc1 shape: {desc1_shape_list}\n"
            error_msg += f"  desc2 shape: {desc2_shape_list}"
            raise ShapeError(
                error_msg,
                actual_shape=dm_shape_list,
                expected_shape=expected_shape,
            )
        return False
    return True


def KORNIA_CHECK_LAF(laf: torch.Tensor, raises: bool = True) -> bool:
    """Check whether a Local Affine Frame (laf) has a valid shape.

    Args:
        laf: local affine frame tensor to evaluate.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        ShapeError: if the input laf does not have a shape :math:`(B,N,2,3)` and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> lafs = torch.rand(2, 10, 2, 3)
        >>> KORNIA_CHECK_LAF(lafs)
        True

    """
    return KORNIA_CHECK_SHAPE(laf, ["B", "N", "2", "3"], raises=raises)


def _handle_invalid_range(
    msg: Optional[str], raises: bool, min_val: float | torch.Tensor, max_val: float | torch.Tensor
) -> bool:
    """Handle invalid range cases."""
    # Extract scalar values if tensors
    min_scalar = min_val.item() if isinstance(min_val, torch.Tensor) else min_val
    max_scalar = max_val.item() if isinstance(max_val, torch.Tensor) else max_val

    err_msg = f"Value range mismatch: expected [0, 1], got [{min_scalar}, {max_scalar}]."
    if msg is not None:
        err_msg += f"\n  {msg}"
    if raises:
        raise ValueCheckError(
            err_msg,
            actual_value=(min_scalar, max_scalar),
            expected_range=(0.0, 1.0),
        )
    return False
