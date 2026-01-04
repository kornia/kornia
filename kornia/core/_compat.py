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


import warnings
from functools import wraps
from inspect import isclass
from typing import Any, Callable, Optional

import torch
from packaging import version


def torch_version() -> str:
    """Parse the PyTorch version string and remove build metadata.

    Extracts the clean version number from `torch.__version__` by removing any
    build suffixes like "+cu118" (CUDA builds) or "+cpu" (CPU builds).

    Returns:
        The clean version string (e.g., "2.0.1" from "2.0.1+cu118").

    Example:
        >>> torch_version()
        '2.0.1'
    """
    return torch.__version__.partition("+")[0]


def torch_version_lt(major: int, minor: int, patch: int) -> bool:
    """Check if the current PyTorch version is less than the specified version.

    Args:
        major: Major version number (e.g., 2 for version 2.x.x).
        minor: Minor version number (e.g., 0 for version x.0.x).
        patch: Patch version number (e.g., 1 for version x.x.1).

    Returns:
        True if the current PyTorch version is strictly less than the specified version,
        False otherwise.

    Example:
        >>> torch_version_lt(2, 0, 0)  # True if PyTorch < 2.0.0
        False
        >>> torch_version_lt(3, 0, 0)  # True if PyTorch < 3.0.0
        True
    """
    _version = version.parse(torch_version())
    return _version < version.parse(f"{major}.{minor}.{patch}")


def torch_version_le(major: int, minor: int, patch: int) -> bool:
    """Check if the current PyTorch version is less than or equal to the specified version.

    Args:
        major: Major version number (e.g., 2 for version 2.x.x).
        minor: Minor version number (e.g., 0 for version x.0.x).
        patch: Patch version number (e.g., 1 for version x.x.1).

    Returns:
        True if the current PyTorch version is less than or equal to the specified version,
        False otherwise.

    Example:
        >>> torch_version_le(2, 0, 0)  # True if PyTorch <= 2.0.0
        True
        >>> torch_version_le(1, 13, 0)  # True if PyTorch <= 1.13.0
        True
    """
    _version = version.parse(torch_version())
    return _version <= version.parse(f"{major}.{minor}.{patch}")


def torch_version_ge(major: int, minor: int, patch: Optional[int] = None) -> bool:
    """Check if the current PyTorch version is greater than or equal to the specified version.

    This function is useful for implementing version-specific behavior or compatibility checks.
    When `patch` is None, only major and minor versions are compared.

    Args:
        major: Major version number (e.g., 2 for version 2.x.x).
        minor: Minor version number (e.g., 0 for version x.0.x).
        patch: Optional patch version number (e.g., 1 for version x.x.1).
            If None, only major.minor version is compared.

    Returns:
        True if the current PyTorch version is greater than or equal to the specified version,
        False otherwise.

    Example:
        >>> torch_version_ge(2, 0, 0)  # True if PyTorch >= 2.0.0
        True
        >>> torch_version_ge(2, 4)  # True if PyTorch >= 2.4.0 (any patch)
        True
        >>> torch_version_ge(3, 0, 0)  # True if PyTorch >= 3.0.0
        False
    """
    _version = version.parse(torch_version())
    if patch is None:
        return _version >= version.parse(f"{major}.{minor}")
    else:
        return _version >= version.parse(f"{major}.{minor}.{patch}")


def _emit_deprecation_warning(
    name: str, replace_with: Optional[str], version: Optional[str], extra_reason: Optional[str]
) -> None:
    """Emit a deprecation warning with the given parameters."""
    beginning = f"Since kornia {version} the " if version is not None else ""
    extra = f" {extra_reason}" if extra_reason else ""

    warnings.simplefilter("always", DeprecationWarning)
    try:
        if replace_with is not None:
            msg = f"{beginning}`{name}` is deprecated in favor of `{replace_with}`.{extra}"
        else:
            msg = f"{beginning}`{name}` is deprecated and will be removed in the future versions.{extra}"
        warnings.warn(
            msg,
            category=DeprecationWarning,
            stacklevel=3,
        )
    finally:
        warnings.simplefilter("default", DeprecationWarning)


def deprecated(
    replace_with: Optional[str] = None, version: Optional[str] = None, extra_reason: Optional[str] = None
) -> Any:
    """Mark functions or classes as deprecated with a warning.

    This decorator emits a :class:`DeprecationWarning` when the decorated function or class is called.
    It provides information about when the deprecation was introduced and what should be used instead.

    Args:
        replace_with: The name of the replacement function/class that should be used instead.
            If provided, the warning message will suggest using this replacement.
        version: The kornia version when the deprecation was introduced (e.g., "0.8.3").
            If provided, the warning message will include "Since kornia {version}".
        extra_reason: Additional context or reason for the deprecation. This will be appended
            to the warning message.

    Returns:
        A decorator that wraps the function or class with deprecation warnings.

    Example:
        Basic usage without replacement:
        ```python
        @deprecated(version="0.8.3")
        def old_function():
            pass
        ```

        With replacement suggestion:
        ```python
        @deprecated(replace_with="new_function", version="0.8.3")
        def old_function():
            pass
        ```

        With additional context:
        ```python
        @deprecated(
            replace_with="new_function",
            version="0.8.3",
            extra_reason=" The old implementation has performance issues."
        )
        def old_function():
            pass
        ```

        Works with classes too:
        ```python
        @deprecated(replace_with="NewClass", version="0.8.3")
        class OldClass:
            pass
        ```
    """

    def _deprecated(func: Callable[..., Any]) -> Any:
        # Get the name - use __name__ for both functions and classes
        name = getattr(func, "__name__", "unknown")

        if isclass(func):
            # For classes, wrap in a function that emits warning and instantiates the class
            # We manually preserve important class attributes since @wraps doesn't work on classes
            def class_wrapper(*args: Any, **kwargs: Any) -> Any:
                _emit_deprecation_warning(name, replace_with, version, extra_reason)
                return func(*args, **kwargs)

            # Preserve important class attributes
            class_wrapper.__name__ = name
            class_wrapper.__qualname__ = getattr(func, "__qualname__", name)
            class_wrapper.__module__ = getattr(func, "__module__", None)
            class_wrapper.__doc__ = func.__doc__
            class_wrapper.__annotations__ = getattr(func, "__annotations__", {})
            return class_wrapper
        else:
            # For functions, use @wraps normally
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                _emit_deprecation_warning(name, replace_with, version, extra_reason)
                return func(*args, **kwargs)

            return wrapper

    return _deprecated
