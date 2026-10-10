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
import pickle
from typing import Any, ClassVar, Optional, Self

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


def _wrappers_in(args: Any, kwargs: Optional[dict[str, Any]] = None) -> list[TensorWrapper]:
    """Return the wrappers among ``args``, one level into a list or tuple, then among the values of ``kwargs``."""
    found: list[TensorWrapper] = []
    for a in (*args, *(kwargs or {}).values()):
        if isinstance(a, TensorWrapper):
            found.append(a)
        elif isinstance(a, collections.abc.Sequence) and not isinstance(a, (str, bytes)):
            found.extend(el for el in a if isinstance(el, TensorWrapper))
    return found


def _result_type(result: Tensor, operands: list[TensorWrapper]) -> Optional[type[TensorWrapper]]:
    """Return the wrapper class a tensor result takes, or ``None`` when it stays a plain tensor.

    The class of an operand with a coordinate width (``Vector2``, ``Vector3``) comes before the class of one without
    (``Scalar``), whichever side of the call either is on. It is taken only when the result still holds such
    vectors: it ends in that width and has at least the operand's number of axes. An operation that removes an axis
    can leave a result that ends in the width by chance (the norms of three vectors), so it returns a plain tensor.
    A class without a width takes any result.
    """
    # ``__class__`` rather than ``type()``: torch 2.5.1's Dynamo cannot construct a class that ``type()`` returned.
    for op in operands:
        width = op.__class__._WIDTH
        if width is not None and op._data.ndim <= result.ndim and result.shape[-1] == width:
            return op.__class__
    for op in operands:
        if op.__class__._WIDTH is None:
            return op.__class__
    return None


def _rewrap(v: Any, operands: list[TensorWrapper]) -> Any:
    """Wrap each tensor in ``v``, or in a list or tuple ``v``, in the class :func:`_result_type` picks."""
    if type(v) in {tuple, list}:
        return type(v)(_rewrap(vi, operands) for vi in v)
    if not isinstance(v, Tensor):
        return v
    cls = _result_type(v, operands)
    return v if cls is None else cls(v)


def _keeps_last_axis(key: Any, ndim: int) -> bool:
    """Return whether indexing a tensor of ``ndim`` axes with ``key`` leaves its last axis last, at most sliced."""
    items = key if isinstance(key, tuple) else (key,)

    def consumed(item: Any) -> int:
        if item is None or isinstance(item, bool):
            return 0
        if isinstance(item, Tensor) and item.dtype in (torch.bool, torch.uint8):
            return item.ndim
        return 1

    ellipsis = [i for i, item in enumerate(items) if item is Ellipsis]
    if ellipsis:
        # The items after the ellipsis index the trailing axes.
        items = items[ellipsis[0] + 1 :]
    elif sum(consumed(item) for item in items) < ndim:
        # The items index leading axes only, and new axes go before the untouched trailing ones.
        return True
    # The last item that consumes an axis indexes the last axis. A new axis after it has size 1, which the caller's
    # width check rejects.
    for item in reversed(items):
        if consumed(item):
            return isinstance(item, slice)
    return True


def _is_picklable(obj: Any) -> bool:
    """Return whether pickle can store ``obj``."""
    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, AttributeError, TypeError):
        return False
    return True


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

    Convention:
        - An attribute the wrapper does not define is looked up on the wrapped tensor, and a tensor value is
          wrapped in the wrapper's class. The wrapper's own names are never forwarded: its slots, ``data``,
          ``__dict__`` and the copy and pickle hooks. So ``copy.copy``, ``copy.deepcopy`` and pickle return the
          wrapper's class, and ``torch.compile`` can trace a function that builds a wrapper. The array and DLPack
          hooks are forwarded, so ``numpy.asarray(w)`` and ``torch.from_dlpack(w)`` convert the wrapped tensor.
        - A torch function called on a wrapper (``torch.clone(w)``), a tensor method reached through the wrapper
          (``w.clone()``), a tensor attribute (``w.T``) and every arithmetic, bitwise and comparison operator
          (``+ - * / // % ** @ & | ^ << >>``, ``== != < <= > >=``, unary ``-``, ``+``, ``abs`` and ``~``) compute
          the wrapped tensors' result and wrap each tensor in it by one rule. The class of an operand with a
          coordinate width (``Vector2``, ``Vector3``) is taken, whichever side it is on, when the result ends in
          that width and keeps at least that operand's axes. Otherwise the class of the first operand without a
          width (``Scalar``, ``TensorWrapper``) is taken: the left operand's (``w + x``), or the right operand's
          when the left one is not a wrapper (``2 / w``, ``t + w``). A comparison whose right operand is an
          instance of a subclass of the left operand's class takes the subclass, because Python tries the
          subclass's reflected comparison first. When no operand qualifies, the result is a plain tensor. So
          ``Scalar * Vector3`` is a ``Vector3``, and ``torch.linalg.norm(v, dim=-1)`` is a plain tensor, also when
          it happens to end in 3. The rule reads shapes only: a result of the same shape stays in the class, as
          the transpose of a ``(3, 3)`` ``Vector3`` does.
        - Indexing returns the indexed tensor in the wrapper's class. For a subclass with a coordinate width, it
          does so only when the key leaves the coordinate axis last, at most sliced, and the result ends in the
          width (``v[0]``, ``v[mask]``); otherwise it returns a plain tensor (``v[..., 0]``, ``v[:, 0]``).
        - An in-place operator (``+=``, ``-=``, ``*=``, ``/=``, ``//=``, ``%=``, ``**=``, ``&=``, ``|=``, ``^=``,
          ``<<=``, ``>>=``) updates the wrapped tensor in place and returns the same wrapper, so an alias sees the
          change. The wrapper does not copy the tensor it is built from, so the update also changes that tensor,
          and it raises where the tensor's in-place operator raises: on a leaf that requires grad, or when the
          result would change the dtype or the shape. ``w @= x`` rebinds ``w`` to ``w @ x``, as ``@=`` does for
          a tensor.
        - ``bool``, ``int``, ``float``, ``complex``, ``operator.index`` and ``len`` return Python values.

    Attributes:
        _data: The underlying PyTorch tensor.
        used_attrs: Set of attribute names that have been accessed.
        used_calls: Set of functions that have been called.
    """

    __slots__ = ("_data", "used_attrs", "used_calls")

    # The size of the last axis a result needs to keep a subclass's class; ``None`` takes any result.
    _WIDTH: ClassVar[Optional[int]] = None

    # Names ``__getattr__`` never forwards to the wrapped tensor: forwarding a slot recurses on an instance whose
    # slots are not set yet (Dynamo builds one while it traces the constructor, and looks up ``__dict__`` on it),
    # and a forwarded copy or pickle hook copies the tensor instead of the wrapper.
    _OWNED_NAMES = frozenset(
        {
            *__slots__,
            "data",
            "__dict__",
            "__copy__",
            "__deepcopy__",
            "__getnewargs__",
            "__getnewargs_ex__",
            "__getstate__",
            "__reduce__",
            "__reduce_ex__",
            "__setstate__",
        }
    )

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
        """Support for pickle serialization.

        Both tracking sets are copied, so a ``copy.copy`` tracks its own usage. ``used_calls`` keeps only the
        functions pickle can store. A few torch functions cannot be pickled, such as ``torch.unique`` or the
        ``Tensor.__pow__`` that ``tensor ** wrapper`` dispatches, and they are left out of the state, so pickling,
        ``torch.save`` and ``copy.deepcopy`` still work after them.
        """
        return {
            "_data": self._data,
            "used_attrs": set(self.used_attrs),
            "used_calls": {func for func in self.used_calls if _is_picklable(func)},
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
        if name in TensorWrapper._OWNED_NAMES:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Track attribute usage
        self.used_attrs.add(name)

        # Get value from underlying tensor
        val = getattr(self._data, name)
        # A tensor method returns a function that wraps the method's result like a torch function's. Deciding
        # this from the class keeps the bound method's type out of it, which Dynamo does not know when the tensor
        # is an intermediate, nor on torch 2.5.1 for any tensor.
        if callable(getattr(Tensor, name, None)):

            def method(*args: Any, **kwargs: Any) -> Any:
                out = val(*_unwrap(args), **{k: _unwrap(v) for k, v in kwargs.items()})
                return _rewrap(out, [self, *_wrappers_in(args, kwargs)])

            return method

        return _rewrap(val, [self])

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

    def __getitem__(self, key: Any) -> Any:
        """Get item from underlying tensor, see the Convention block."""
        out = self._data[key]
        width = self.__class__._WIDTH
        if width is None or (_keeps_last_axis(key, self._data.ndim) and out.ndim >= 1 and out.shape[-1] == width):
            return self.__class__(out)
        return out

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

        operands = _wrappers_in(args, kwargs)

        # Track function usage
        for a in operands:
            if isinstance(a, cls):
                a.used_calls.add(func)

        # Unwrap arguments and call the function
        unwrapped_args = _unwrap(args)
        unwrapped_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}

        return _rewrap(func(*unwrapped_args, **unwrapped_kwargs), operands)

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

    def __rtruediv__(self, other: Any) -> TensorWrapper:
        """Right-side true division operation."""
        return self.__binary_op__(Tensor.__rtruediv__, other)

    def __floordiv__(self, other: Any) -> TensorWrapper:
        """Floor division operation."""
        return self.__binary_op__(torch.floor_divide, other)

    def __rfloordiv__(self, other: Any) -> TensorWrapper:
        """Right-side floor division operation."""
        return self.__binary_op__(Tensor.__rfloordiv__, other)

    def __mod__(self, other: Any) -> TensorWrapper:
        """Remainder operation."""
        return self.__binary_op__(Tensor.__mod__, other)

    def __rmod__(self, other: Any) -> TensorWrapper:
        """Right-side remainder operation."""
        return self.__binary_op__(Tensor.__rmod__, other)

    def __pow__(self, other: Any) -> TensorWrapper:
        """Power operation."""
        # ``torch.pow`` rather than ``Tensor.__pow__``, which pickle cannot store in ``used_calls``. Like
        # ``Tensor.__pow__``, an operand ``torch.pow`` rejects returns NotImplemented, so Python tries its ``__rpow__``.
        try:
            return self.__binary_op__(torch.pow, other)
        except TypeError:
            return NotImplemented

    def __rpow__(self, other: Any) -> TensorWrapper:
        """Right-side power operation."""
        return self.__binary_op__(Tensor.__rpow__, other)

    def __matmul__(self, other: Any) -> TensorWrapper:
        """Matrix multiplication operation."""
        return self.__binary_op__(Tensor.__matmul__, other)

    def __rmatmul__(self, other: Any) -> TensorWrapper:
        """Right-side matrix multiplication operation."""
        return self.__binary_op__(Tensor.__rmatmul__, other)

    def __and__(self, other: Any) -> TensorWrapper:
        """Bitwise and operation."""
        return self.__binary_op__(Tensor.__and__, other)

    def __rand__(self, other: Any) -> TensorWrapper:
        """Right-side bitwise and operation."""
        return self.__binary_op__(Tensor.__rand__, other)

    def __or__(self, other: Any) -> TensorWrapper:
        """Bitwise or operation."""
        return self.__binary_op__(Tensor.__or__, other)

    def __ror__(self, other: Any) -> TensorWrapper:
        """Right-side bitwise or operation."""
        return self.__binary_op__(Tensor.__ror__, other)

    def __xor__(self, other: Any) -> TensorWrapper:
        """Bitwise exclusive or operation."""
        return self.__binary_op__(Tensor.__xor__, other)

    def __rxor__(self, other: Any) -> TensorWrapper:
        """Right-side bitwise exclusive or operation."""
        return self.__binary_op__(Tensor.__rxor__, other)

    def __lshift__(self, other: Any) -> TensorWrapper:
        """Left shift operation."""
        return self.__binary_op__(Tensor.__lshift__, other)

    def __rlshift__(self, other: Any) -> TensorWrapper:
        """Right-side left shift operation."""
        return self.__binary_op__(Tensor.__rlshift__, other)

    def __rshift__(self, other: Any) -> TensorWrapper:
        """Right shift operation."""
        return self.__binary_op__(Tensor.__rshift__, other)

    def __rrshift__(self, other: Any) -> TensorWrapper:
        """Right-side right shift operation."""
        return self.__binary_op__(Tensor.__rrshift__, other)

    def __iadd__(self, other: Any) -> Self:
        """In-place add operation."""
        return self.__inplace_op__(Tensor.__iadd__, other)

    def __isub__(self, other: Any) -> Self:
        """In-place subtract operation."""
        return self.__inplace_op__(Tensor.__isub__, other)

    def __imul__(self, other: Any) -> Self:
        """In-place multiply operation."""
        return self.__inplace_op__(Tensor.__imul__, other)

    def __itruediv__(self, other: Any) -> Self:
        """In-place true division operation."""
        return self.__inplace_op__(Tensor.__itruediv__, other)

    def __ifloordiv__(self, other: Any) -> Self:
        """In-place floor division operation."""
        return self.__inplace_op__(Tensor.__ifloordiv__, other)

    def __imod__(self, other: Any) -> Self:
        """In-place remainder operation."""
        return self.__inplace_op__(Tensor.__imod__, other)

    def __ipow__(self, other: Any) -> Self:
        """In-place power operation."""
        return self.__inplace_op__(Tensor.pow_, other)

    def __iand__(self, other: Any) -> Self:
        """In-place bitwise and operation."""
        return self.__inplace_op__(Tensor.__iand__, other)

    def __ior__(self, other: Any) -> Self:
        """In-place bitwise or operation."""
        return self.__inplace_op__(Tensor.__ior__, other)

    def __ixor__(self, other: Any) -> Self:
        """In-place bitwise exclusive or operation."""
        return self.__inplace_op__(Tensor.__ixor__, other)

    def __ilshift__(self, other: Any) -> Self:
        """In-place left shift operation."""
        return self.__inplace_op__(Tensor.__ilshift__, other)

    def __irshift__(self, other: Any) -> Self:
        """In-place right shift operation."""
        return self.__inplace_op__(Tensor.__irshift__, other)

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

    def __float__(self) -> float:
        """Convert to float (unwrapped)."""
        return float(self._data)

    def __complex__(self) -> complex:
        """Convert to complex (unwrapped)."""
        return complex(self._data)

    def __index__(self) -> int:
        """Convert to an index (unwrapped)."""
        return self._data.__index__()

    def __neg__(self) -> TensorWrapper:
        """Negation operation."""
        return self.__unary_op__(torch.neg)

    def __pos__(self) -> TensorWrapper:
        """Unary plus operation."""
        return self.__unary_op__(Tensor.__pos__)

    def __abs__(self) -> TensorWrapper:
        """Absolute value operation."""
        return self.__unary_op__(Tensor.__abs__)

    def __invert__(self) -> TensorWrapper:
        """Bitwise not operation."""
        return self.__unary_op__(Tensor.__invert__)

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

    def __inplace_op__(self, func: Any, other: Any) -> Self:
        """Helper for in-place operations.

        Args:
            func: The in-place tensor operator to call, such as ``Tensor.__iadd__``.
            other: The other operand.

        Returns:
            This wrapper, whose tensor ``func`` updated in place. Whatever ``func`` raises propagates.
        """
        self.used_calls.add(func)
        func(self._data, _unwrap(other))
        return self

    def __unary_op__(self, func: Any) -> TensorWrapper:
        """Helper for unary operations.

        Args:
            func: The PyTorch function to call.

        Returns:
            Wrapped result of the operation.
        """
        return self.__torch_function__(func, (type(self),), (self,))
