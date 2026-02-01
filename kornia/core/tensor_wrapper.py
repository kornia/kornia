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

from __future__ import annotations

import collections.abc
from typing import Any, Optional

import torch
from torch import Tensor

from kornia.core.check import KORNIA_CHECK_IS_TENSOR


def _wrap(v: Any, cls: type[TensorWrapper]) -> Any:
    """Wrap type.

    Args:
        v: Value to wrap (tensor, list, tuple, or other).
        cls: TensorWrapper class to use for wrapping.

    Returns:
        Wrapped value if tensor, otherwise original value or wrapped collection.
    """
    # wrap inputs if necessary
    if type(v) in {tuple, list}:
        return type(v)(_wrap(vi, cls) for vi in v)

    return cls(v) if isinstance(v, Tensor) else v


def _unwrap(v: Any) -> Any:
    """Unwrap nested type.

    Args:
        v: Value to unwrap (TensorWrapper, list, tuple, or other).

    Returns:
        Unwrapped value (underlying tensor or original value).
    """
    if type(v) in {tuple, list}:
        return type(v)(_unwrap(vi) for vi in v)

    return v._data if isinstance(v, TensorWrapper) else v


class TensorWrapper:
    """Wrapper around PyTorch tensors that tracks attribute and function usage.

    This class provides a transparent wrapper around PyTorch tensors while
    tracking which attributes and functions are accessed. Useful for debugging
    and understanding tensor usage patterns.

    Attributes:
        _data: The underlying PyTorch tensor.
        used_attrs: Set of attribute names that have been accessed.
        used_calls: Set of functions that have been called.
    """

    __slots__ = ("_data", "used_attrs", "used_calls")

    def __init__(self, data: Tensor) -> None:
        """Initialize TensorWrapper with a PyTorch tensor.

        Args:
            data: The PyTorch tensor to wrap. If data is already a TensorWrapper,
                its underlying tensor will be extracted.

        Raises:
            TypeCheckError: If data is not a PyTorch tensor or TensorWrapper.
        """
        # Handle case where data is already a TensorWrapper (e.g., Scalar wrapping another Scalar)
        if isinstance(data, TensorWrapper):
            data = data._data
        KORNIA_CHECK_IS_TENSOR(data, "Expected Tensor for TensorWrapper")
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "used_attrs", set())
        object.__setattr__(self, "used_calls", set())

    def unwrap(self) -> Tensor:
        """Return the underlying PyTorch tensor."""
        return _unwrap(self)

    @property
    def data(self) -> Tensor:
        """Access the underlying tensor."""
        return self._data

    def __getstate__(self) -> dict[str, Any]:
        """Support for pickle serialization."""
        return {
            "_data": self._data,
            "used_attrs": self.used_attrs,
            "used_calls": self.used_calls,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Support for pickle deserialization."""
        object.__setattr__(self, "_data", state["_data"])
        object.__setattr__(self, "used_attrs", state.get("used_attrs", set()))
        object.__setattr__(self, "used_calls", state.get("used_calls", set()))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}({self._data})"

    def __getattr__(self, name: str) -> Any:
        """Get attribute from underlying tensor."""
        # Handle special 'data' property
        if name == "data":
            return self._data

        # Track attribute usage
        self.used_attrs.add(name)

        # Get value from underlying tensor
        val = getattr(self._data, name)

        # Wrap the result if it's a tensor
        return _wrap(val, type(self))

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute on underlying tensor."""
        # Only track non-internal attributes
        if name not in self.__slots__:
            self.used_attrs.add(name)
            setattr(self._data, name, value)
        else:
            # Use object.__setattr__ for internal attributes to avoid recursion
            object.__setattr__(self, name, value)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set item on underlying tensor."""
        self._data[key] = value

    def __getitem__(self, key: Any) -> TensorWrapper:
        """Get item from underlying tensor."""
        return _wrap(self._data[key], type(self))

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Intercept PyTorch function calls."""
        if kwargs is None:
            kwargs = {}

        # Find instances of this class in the arguments
        args_of_this_cls: list[TensorWrapper] = []
        for a in args:
            if isinstance(a, cls):
                args_of_this_cls.append(a)
            elif isinstance(a, collections.abc.Sequence) and not isinstance(a, str | bytes):
                args_of_this_cls.extend(el for el in a if isinstance(el, cls))

        # Track function usage
        for a in args_of_this_cls:
            a.used_calls.add(func)

        # Unwrap arguments and call the function
        unwrapped_args = _unwrap(args)
        unwrapped_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}

        return _wrap(func(*unwrapped_args, **unwrapped_kwargs), cls)

    def __add__(self, other: Any) -> TensorWrapper:
        """Add operation."""
        return self.__binary_op__(torch.add, other)

    def __radd__(self, other: Any) -> TensorWrapper:
        """Right-side add operation."""
        return self.__binary_op__(torch.add, other, swap=True)

    def __mul__(self, other: Any) -> TensorWrapper:
        """Multiply operation."""
        return self.__binary_op__(torch.mul, other)

    def __rmul__(self, other: Any) -> TensorWrapper:
        """Right-side multiply operation."""
        return self.__binary_op__(torch.mul, other, swap=True)

    def __sub__(self, other: Any) -> TensorWrapper:
        """Subtract operation."""
        return self.__binary_op__(torch.sub, other)

    def __rsub__(self, other: Any) -> TensorWrapper:
        """Right-side subtract operation."""
        return self.__binary_op__(torch.sub, other, swap=True)

    def __truediv__(self, other: Any) -> TensorWrapper:
        """True division operation."""
        return self.__binary_op__(torch.true_divide, other)

    def __floordiv__(self, other: Any) -> TensorWrapper:
        """Floor division operation."""
        return self.__binary_op__(torch.floor_divide, other)

    def __ge__(self, other: Any) -> TensorWrapper:
        """Greater than or equal comparison."""
        return self.__binary_op__(torch.ge, other)

    def __gt__(self, other: Any) -> TensorWrapper:
        """Greater than comparison."""
        return self.__binary_op__(torch.gt, other)

    def __lt__(self, other: Any) -> TensorWrapper:
        """Less than comparison."""
        return self.__binary_op__(torch.lt, other)

    def __le__(self, other: Any) -> TensorWrapper:
        """Less than or equal comparison."""
        return self.__binary_op__(torch.le, other)

    def __eq__(self, other: object) -> TensorWrapper:
        """Equality comparison."""
        return self.__binary_op__(torch.eq, other)

    def __ne__(self, other: object) -> TensorWrapper:
        """Inequality comparison."""
        return self.__binary_op__(torch.ne, other)

    def __bool__(self) -> bool:
        """Convert to boolean (unwrapped)."""
        return bool(self._data)

    def __int__(self) -> int:
        """Convert to integer (unwrapped)."""
        return int(self._data)

    def __neg__(self) -> TensorWrapper:
        """Negation operation."""
        return self.__unary_op__(torch.neg)

    def __len__(self) -> int:
        """Return length of tensor."""
        return len(self._data)

    def __binary_op__(self, func: Any, other: Any, swap: bool = False) -> TensorWrapper:
        """Helper for binary operations.

        Args:
            func: The PyTorch function to call.
            other: The other operand.
            swap: If True, swap the order of operands (for right-side operations).

        Returns:
            Wrapped result of the operation.
        """
        if swap:
            args = (other, self)
        else:
            args = (self, other)
        return self.__torch_function__(func, (type(self),), args)

    def __unary_op__(self, func: Any) -> TensorWrapper:
        """Helper for unary operations.

        Args:
            func: The PyTorch function to call.

        Returns:
            Wrapped result of the operation.
        """
        return self.__torch_function__(func, (type(self),), (self,))
