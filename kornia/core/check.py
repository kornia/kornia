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

"""Validate inputs: check conditions, shapes, sizes, types, devices and image layouts.

Every ``KORNIA_CHECK*`` function returns ``True`` when its check passes, and, outside TorchScript, also without
checking while the checks are disabled. On failure it raises an error from ``kornia.core.exceptions``, or returns
``False`` with ``raises=False``. A ``msg`` argument, where a check takes one, is a string or a function without
arguments that returns it, called only on failure; see :func:`KORNIA_CHECK`. To turn the checks off, see
:func:`disable_checks`.

Which function to use:

- Any boolean condition: :func:`KORNIA_CHECK`, which raises the plain :exc:`~kornia.core.exceptions.BaseError`.
- Shape: against a pattern of named and fixed dimensions such as ``["B", "3", "H", "W"]``, :func:`KORNIA_CHECK_SHAPE`;
  against sizes computed at run time, :func:`KORNIA_CHECK_SIZES`; one dimension against another tensor's,
  :func:`KORNIA_CHECK_SAME_SIZE`; every dimension against another tensor's, :func:`KORNIA_CHECK_SAME_SHAPE`; a distance
  matrix against its descriptors, :func:`KORNIA_CHECK_DM_DESC`; the shape of local affine frames,
  :func:`~kornia.feature.KORNIA_CHECK_LAF`.
- Image layout: color, :func:`KORNIA_CHECK_IS_COLOR`; gray, :func:`KORNIA_CHECK_IS_GRAY`; either,
  :func:`KORNIA_CHECK_IS_COLOR_OR_GRAY`; either, with values in :math:`[0, 1]`, or up to ``2 ** bits - 1`` for integers,
  :func:`KORNIA_CHECK_IS_IMAGE`.
- Type: a tensor, :func:`KORNIA_CHECK_IS_TENSOR`; any type, :func:`KORNIA_CHECK_TYPE`; a list of tensors,
  :func:`KORNIA_CHECK_IS_LIST_OF_TENSOR`; narrow an optional value for a static type checker, without checking it,
  :func:`KORNIA_UNWRAP`.
- Device: two tensors, :func:`KORNIA_CHECK_SAME_DEVICE`; a list of tensors, :func:`KORNIA_CHECK_SAME_DEVICES`.
- Turn the checks off and on: :func:`disable_checks`, :func:`enable_checks`; read the state: :func:`are_checks_enabled`.
- The errors, re-exported from ``kornia.core.exceptions``: :exc:`~kornia.core.exceptions.BaseError`,
  :exc:`~kornia.core.exceptions.ShapeError`, :exc:`~kornia.core.exceptions.TypeCheckError`,
  :exc:`~kornia.core.exceptions.ValueCheckError`, :exc:`~kornia.core.exceptions.AsyncCheckError`,
  :exc:`~kornia.core.exceptions.DeviceError`, :exc:`~kornia.core.exceptions.ImageError`.
"""

from __future__ import annotations

import os
from types import UnionType
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, TypeVar, Union, cast, get_args, get_origin

import torch
from torch import float16, float32, float64

from kornia.core._compat import torch_version_ge
from kornia.core.exceptions import (
    AsyncCheckError,
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
    "KORNIA_CHECK_IS_COLOR_OR_GRAY",
    "KORNIA_CHECK_IS_GRAY",
    "KORNIA_CHECK_IS_IMAGE",
    "KORNIA_CHECK_IS_LIST_OF_TENSOR",
    "KORNIA_CHECK_IS_TENSOR",
    "KORNIA_CHECK_LAF",
    "KORNIA_CHECK_SAME_DEVICE",
    "KORNIA_CHECK_SAME_DEVICES",
    "KORNIA_CHECK_SAME_SHAPE",
    "KORNIA_CHECK_SAME_SIZE",
    "KORNIA_CHECK_SHAPE",
    "KORNIA_CHECK_SIZES",
    "KORNIA_CHECK_TYPE",
    "KORNIA_UNWRAP",
    "AsyncCheckError",
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

# ``aten::_assert_async.msg`` gained its MPS kernel in torch 2.13; kornia supports torch >= 2.5.1.
_MPS_HAS_ASSERT_ASYNC = torch_version_ge(2, 13)

if TYPE_CHECKING:
    # A check message is a string, or a function without arguments that builds it; see ``KORNIA_CHECK``.
    CheckMessage = Union[str, Callable[[], str]]
else:
    # TorchScript resolves the annotations at run time and has no type for a Python function, so it sees ``str``. A
    # message function is Python-only, and ``_resolve_msg`` calls it under a bare ``is_scripting()`` guard.
    CheckMessage = str


def _resolve_msg(msg: Optional[CheckMessage]) -> Optional[str]:
    """Return the message of a failed check, calling it first if it is a function."""
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if callable(msg):
            return msg()
    return msg


def are_checks_enabled() -> bool:
    """Report whether kornia's input checks currently run.

    ``enabled = are_checks_enabled()`` returns the flag that :func:`disable_checks` and :func:`enable_checks` set. At
    import, the ``KORNIA_CHECKS`` environment variable sets it: ``1``, ``true``, ``yes`` or ``on``, in any case, enable
    the checks, and any other value, the empty string included, disables them. Without the variable, checks are enabled,
    except under ``python -O``.

    Returns:
        ``True`` if checks are enabled, ``False`` otherwise.

    See Also:
        - :func:`disable_checks`: turns the checks off.
        - :func:`enable_checks`: turns them back on.

    Example:
        >>> are_checks_enabled()
        True
    """
    return _KORNIA_CHECKS_ENABLED


def disable_checks() -> None:
    """Turn off kornia's input checks until enable_checks turns them back on.

    After ``disable_checks()``, every ``KORNIA_CHECK*`` function returns ``True`` without checking, in eager code and
    under ``torch.compile``. The value checks behind :exc:`~kornia.core.exceptions.AsyncCheckError` are skipped too,
    except the input-range checks of :func:`~kornia.enhance.equalize` and :func:`~kornia.enhance.equalize_clahe`, which
    keep a lookup index in bounds. A call compiled with ``torch.jit.script`` ignores the flag and runs the
    ``KORNIA_CHECK*`` functions regardless; on CUDA it skips the value checks behind
    :exc:`~kornia.core.exceptions.AsyncCheckError`, enabled or not.

    See Also:
        - :func:`enable_checks`: turns the checks back on.
        - :func:`are_checks_enabled`: reads the current state.

    Example:
        >>> disable_checks()
        >>> are_checks_enabled()
        False
    """
    global _KORNIA_CHECKS_ENABLED  # noqa: PLW0603
    _KORNIA_CHECKS_ENABLED = False


def enable_checks() -> None:
    """Turn kornia's input checks back on, also after KORNIA_CHECKS=0 or python -O.

    After ``enable_checks()``, every check runs again.

    See Also:
        - :func:`disable_checks`: turns the checks off.
        - :func:`are_checks_enabled`: reads the current state.

    Example:
        >>> enable_checks()
        >>> are_checks_enabled()
        True
    """
    global _KORNIA_CHECKS_ENABLED  # noqa: PLW0603
    _KORNIA_CHECKS_ENABLED = True


# Logger api


def KORNIA_CHECK_SHAPE(
    x: torch.Tensor, shape: list[str], msg: Optional[CheckMessage] = None, raises: bool = True
) -> bool:
    """Check a tensor's shape against a pattern of named and fixed dimensions.

    ``KORNIA_CHECK_SHAPE(x, ["B", "3", "H", "W"])`` returns ``True`` when ``x`` has one dimension per entry and every
    entry made of digits equals the size of its dimension. Any other entry, such as ``"B"``, matches any size, and equal
    names are not compared with each other: ``["N", "N"]`` accepts a ``(2, 3)`` tensor. A ``"*"`` as the first entry
    matches any number of leading dimensions; otherwise a ``"*"`` as the last entry matches any number of trailing ones.
    It does not check that ``x`` is a tensor.

    Args:
        x: the tensor to evaluate.
        shape: the expected pattern, one string per dimension, with a ``"*"`` allowed first or last: ``["*"]`` accepts
            any shape and ``[]`` only a 0-dimensional tensor.
        msg: message to append to the error, or a function that returns it; see :func:`KORNIA_CHECK`.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        ShapeError: if ``x`` does not match ``shape`` and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_SIZES`: takes integer sizes computed at run time, ``None`` for any size.
        - :func:`KORNIA_CHECK_SAME_SHAPE`: compares the whole shape with another tensor's.
        - :func:`~kornia.feature.KORNIA_CHECK_LAF`: this check with the fixed pattern ``["B", "N", "2", "3"]``.

    Example:
        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["B", "C", "H", "W"])
        True

        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["2", "3", "H", "W"])
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not _KORNIA_CHECKS_ENABLED:
            return True

    dimension_offset = 0
    if len(shape) > 0 and "*" == shape[0]:
        shape_to_check = shape[1:]
        dimension_offset = max(len(x.shape) - len(shape_to_check), 0)
        x_shape_to_check = x.shape[dimension_offset:]
    elif len(shape) > 0 and "*" == shape[-1]:
        shape_to_check = shape[:-1]
        x_shape_to_check = x.shape[: len(shape) - 1]
    else:
        shape_to_check = shape
        x_shape_to_check = x.shape

    if len(x_shape_to_check) != len(shape_to_check):
        if raises:
            msg = _resolve_msg(msg)
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
                msg = _resolve_msg(msg)
                error_msg = (
                    f"Shape mismatch at dimension {i + dimension_offset}: expected {dim}, got {x_shape_to_check[i]}.\n"
                )
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
            return False
    return True


def KORNIA_CHECK(condition: bool, msg: Optional[CheckMessage] = None, raises: bool = True) -> bool:
    """Check an arbitrary boolean condition, raising the base kornia error when it is false.

    ``KORNIA_CHECK(condition, msg)`` returns ``True`` when ``condition`` is true. Otherwise it raises
    :exc:`~kornia.core.exceptions.BaseError` with ``msg`` as the message (``"Validation condition failed"`` without
    one). :exc:`~kornia.core.exceptions.BaseError` is neither a :exc:`ValueError` nor a :exc:`TypeError`; for a shape, a
    size, a type or a device, use the dedicated check, whose error is one of them.

    Args:
        condition: the condition to evaluate.
        msg: message to show in the exception, or a function without arguments that returns it. A function is called
            only when the check fails. Python builds an f-string argument on every call, and under a dynamic-shape
            ``torch.compile`` formatting a tensor size or an ``int`` argument specializes it, so each new value would
            compile a new graph. A check on sizes therefore compares them with :func:`KORNIA_CHECK_SAME_SIZE`,
            :func:`KORNIA_CHECK_SIZES` or :func:`KORNIA_CHECK_SHAPE`, which format only on failure, and any other
            message that formats such a value is a function, as in ``lambda: f"got {n}"``. TorchScript takes only a
            string.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        BaseError: if the condition is not met and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_SHAPE`: checks a shape pattern and raises :exc:`~kornia.core.exceptions.ShapeError`.
        - :func:`KORNIA_CHECK_SIZES`: checks sizes computed at run time without specializing them.
        - :func:`KORNIA_CHECK_TYPE`: checks a type and raises :exc:`~kornia.core.exceptions.TypeCheckError`.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK(x.shape[-2:] == (3, 3), "Invalid homography")
        True
        >>> KORNIA_CHECK(x.shape[0] == 2, lambda: f"Expected a batch of 2, got {x.shape[0]}")
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if not condition:
        if raises:
            msg = _resolve_msg(msg)
            error_msg = "Validation condition failed" if msg is None else msg
            raise BaseError(error_msg)
        return False
    return True


def _vmap_is_active() -> bool:
    """Whether a ``torch.vmap`` level is on the functorch transform stack."""
    stack = torch._C._functorch.get_interpreter_stack()
    return stack is not None and any(level.key() == torch._C._functorch.TransformType.Vmap for level in stack)


def _check_async(condition: torch.Tensor, msg: str, guard: bool = False) -> None:
    """Check a boolean tensor ``condition`` where it lives, without reading it back from an accelerator.

    On CUDA, and on MPS from torch 2.13, the check is ``torch._assert_async``: the call returns at once, and a false
    ``condition`` raises :exc:`RuntimeError` by the next synchronization, possibly earlier: on CUDA often at the next
    kernel launch, still inside the calling function. On CUDA it is a device-side assert, after which the context
    cannot be used. Reading the condition on the host instead would wait for every queued kernel.

    On the CPU, on MPS before torch 2.13, which has no assert kernel, and on any other device, the condition is read on
    the host and a false one raises :exc:`~kornia.core.exceptions.AsyncCheckError` during the call. A meta tensor has
    no value to check.

    Under ``torch.compile`` and ``torch.export`` the check is ``torch._assert_async`` on every device that has the
    kernel, so it stays in the graph and raises :exc:`RuntimeError`; Inductor keeps it from torch 2.9 and compiles it
    to nothing before. On a device without the kernel nothing is checked there. ``torch.jit.trace`` runs the assert on
    the traced values but drops it from the graph it records, and TorchScript drops it too, so a scripted call checks
    on the host only. It does so on every device but CUDA, since TorchScript cannot read whether MPS has the kernel.
    A scripted CUDA call is not checked. The check is skipped when checks are disabled with :func:`disable_checks`,
    with ``python -O`` or with ``KORNIA_CHECKS=0``, unless ``guard`` is set. It is also skipped under ``torch.vmap``,
    which has no batching rule for the assert. Eager calls keep it under the other functorch transforms, such as
    ``torch.func.grad``; a compiled call skips it under all of them, unless ``guard`` is set, since Dynamo cannot
    inspect the transform stack.

    Args:
        condition: a boolean tensor with one element, usually ``(...).all()``.
        msg: the message of the error. It is a string, since the device assert carries it unformatted.
        guard: run the check even when checks are disabled, and under a compiled functorch transform. Set it where the
            check keeps a later index in bounds: on the CPU a compiled ``gather`` or ``scatter`` with an index out of
            range terminates the process. A compiled ``torch.vmap`` with ``fullgraph=True`` then fails to trace, as the
            assert has no batching rule, so set it only in code that cannot run under ``vmap`` anyway, such as the
            in-place histograms of the ``equalize`` lookups.
    """
    # TorchScript cannot read ``_MPS_HAS_ASSERT_ASYNC``, so a scripted call reads MPS and other devices on the host.
    on_host = condition.device.type not in ("cuda", "meta")
    if not torch.jit.is_scripting():
        if not (guard or _KORNIA_CHECKS_ENABLED):
            return
        if torch._C._are_functorch_transforms_active():
            if torch.compiler.is_compiling():
                # Dynamo cannot trace the stack walk, so a compiled call skips the check under every functorch
                # transform. A guard stays: it keeps an index in bounds, and the assert works under grad.
                if not guard:
                    return
            elif _vmap_is_active():
                return
        kind = condition.device.type
        has_kernel = kind in ("cpu", "cuda", "meta") or (kind == "mps" and _MPS_HAS_ASSERT_ASYNC)
        if torch.compiler.is_compiling() or torch.jit.is_tracing():
            if has_kernel:
                torch._assert_async(condition, msg)
            return
        on_host = not has_kernel or kind == "cpu"
    if on_host:
        if not bool(condition):
            raise AsyncCheckError(msg)
    else:
        torch._assert_async(condition, msg)


def KORNIA_UNWRAP(maybe_obj: Any, typ: Any) -> Any:
    """Narrow an optional value to a type for static type checkers, without checking it.

    ``obj = KORNIA_UNWRAP(maybe_obj, typ)`` returns ``maybe_obj`` unchanged: it is :func:`typing.cast`, so ``None`` and
    a value of another type pass through, whether checks are enabled or not. To check the type at run time, use
    :func:`KORNIA_CHECK_TYPE`. TorchScript cannot compile it.

    Args:
        maybe_obj: the object to unwrap.
        typ: the type that static type checkers assume for the result.

    Returns:
        ``maybe_obj``, unchanged.

    See Also:
        - :func:`KORNIA_CHECK_TYPE`: raises :exc:`~kornia.core.exceptions.TypeCheckError`.

    """
    # TODO: this function will change after kornia/pr#1987
    return cast(typ, maybe_obj)


_T = TypeVar("_T", bound=type)


def _type_name(typ: Any) -> str:
    """Name a type for an error message, spelling a union of any kind as ``int | str``."""
    # A union's own name says nothing: typing.Union[...] is "Union", and from Python 3.14 on so is ``int | str``, which
    # has no ``__name__`` before 3.14. Name it from its members instead.
    if get_origin(typ) in (Union, UnionType):
        return " | ".join("None" if arg is type(None) else _type_name(arg) for arg in get_args(typ))
    return getattr(typ, "__name__", repr(typ))


def KORNIA_CHECK_TYPE(
    x: Any, typ: _T | tuple[_T, ...], msg: Optional[CheckMessage] = None, raises: bool = True
) -> bool:
    """Check that a value is an instance of a type, a tuple of types or a union.

    ``KORNIA_CHECK_TYPE(x, typ)`` returns ``True`` when ``isinstance(x, typ)``, with ``typ`` a class, a tuple of classes
    or a union such as ``int | str``; as in :func:`isinstance`, ``True`` passes for ``int``. On failure it raises
    :exc:`~kornia.core.exceptions.TypeCheckError`. TorchScript cannot compile it.

    Args:
        x: any input variable.
        typ: the expected type, a tuple of types or a union.
        msg: message to show in the exception, or a function that returns it; see :func:`KORNIA_CHECK`.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        TypeCheckError: if ``x`` is not an instance of ``typ`` and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_IS_TENSOR`: checks for a tensor, and TorchScript compiles it.
        - :func:`KORNIA_CHECK_IS_LIST_OF_TENSOR`: checks a list and the type of each of its elements.

    Example:
        >>> KORNIA_CHECK_TYPE("foo", str, "Invalid string")
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if not isinstance(x, typ):
        if raises:
            msg = _resolve_msg(msg)
            # JIT doesn't support try-except or type introspection, so use simple message
            if torch.jit.is_scripting():
                error_msg = "Type mismatch."
                if msg is not None:
                    error_msg += f"\n  {msg}"
                raise TypeCheckError(error_msg)
            # In Python mode, we can safely use type introspection
            expected_type_str = " | ".join(_type_name(t) for t in typ) if isinstance(typ, tuple) else _type_name(typ)
            type_name = str(type(x))
            error_msg = f"Type mismatch: expected {expected_type_str}, got {type_name}."
            if msg is not None:
                error_msg += f"\n  {msg}"
            raise TypeCheckError(
                error_msg,
                actual_type=type(x),
                expected_type=typ,
            )
        return False
    return True


def KORNIA_CHECK_IS_TENSOR(x: Any, msg: Optional[CheckMessage] = None, raises: bool = True) -> bool:
    """Check that a value is a torch.Tensor, a subclass such as a Parameter included.

    ``KORNIA_CHECK_IS_TENSOR(x)`` returns ``True`` when ``isinstance(x, torch.Tensor)``, so a ``torch.nn.Parameter``
    passes and a NumPy array fails. On failure it raises :exc:`~kornia.core.exceptions.TypeCheckError`.

    Args:
        x: any input variable.
        msg: message to show in the exception, or a function that returns it; see :func:`KORNIA_CHECK`.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        TypeCheckError: if ``x`` is not a tensor and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_TYPE`: checks against any type, tuple of types or union.
        - :func:`KORNIA_CHECK_IS_LIST_OF_TENSOR`: checks a list whose elements are tensors.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_IS_TENSOR(x, "Invalid tensor")
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if not isinstance(x, torch.Tensor):
        if raises:
            msg = _resolve_msg(msg)
            # JIT doesn't support try-except or type introspection, so use simple message
            if torch.jit.is_scripting():
                error_msg = "Type mismatch: expected Tensor."
                if msg is not None:
                    error_msg += f"\n  {msg}"
                raise TypeCheckError(error_msg)

            # In Python mode, we can safely use type introspection
            type_name = str(type(x))
            error_msg = f"Type mismatch: expected Tensor, got {type_name}."
            if msg is not None:
                error_msg += f"\n  {msg}"
            raise TypeCheckError(
                error_msg,
                actual_type=type(x),
                expected_type=torch.Tensor,
            )
        return False
    return True


def KORNIA_CHECK_IS_LIST_OF_TENSOR(x: Optional[Sequence[Any]], raises: bool = True) -> bool:
    """Check that a value is a Python list whose elements are all tensors.

    ``KORNIA_CHECK_IS_LIST_OF_TENSOR(x)`` returns ``True`` for a ``list`` of tensors, the empty list included. A tuple,
    ``None`` or a list with an element that is not a tensor raises :exc:`~kornia.core.exceptions.TypeCheckError`.
    TorchScript cannot compile it.

    Args:
        x: any sequence of objects.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        TypeCheckError: if ``x`` is not a list of tensors and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_SAME_DEVICES`: also requires a non-empty list, and that its tensors share a device.
        - :func:`KORNIA_CHECK_IS_TENSOR`: checks a single value.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_IS_LIST_OF_TENSOR(x, raises=False)
        False
        >>> KORNIA_CHECK_IS_LIST_OF_TENSOR([x])
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not _KORNIA_CHECKS_ENABLED:
            return True

    are_tensors = isinstance(x, list) and all(isinstance(d, torch.Tensor) for d in x)
    if not are_tensors:
        if raises:
            got = type(x).__name__
            if isinstance(x, list):
                index = next(i for i, d in enumerate(x) if not isinstance(d, torch.Tensor))
                got += f"; element {index} is {type(x[index]).__name__}"
            error_msg = f"Type mismatch: expected list[Tensor], got {got}."
            raise TypeCheckError(
                error_msg,
                actual_type=type(x),
                expected_type=list[torch.Tensor],
            )
        return False
    return True


def KORNIA_CHECK_SAME_DEVICE(x: torch.Tensor, y: torch.Tensor, raises: bool = True) -> bool:
    """Check that two tensors are on the same device.

    ``KORNIA_CHECK_SAME_DEVICE(x, y)`` returns ``True`` when ``x.device == y.device``. On failure it raises
    :exc:`~kornia.core.exceptions.DeviceError`.

    Args:
        x: first tensor to evaluate.
        y: second tensor to evaluate.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        DeviceError: if the two tensors are not on the same device and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_SAME_DEVICES`: checks a list of tensors against the first one.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(1, 3, 1)
        >>> KORNIA_CHECK_SAME_DEVICE(x1, x2)
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
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


def KORNIA_CHECK_SAME_DEVICES(
    tensors: list[torch.Tensor], msg: Optional[CheckMessage] = None, raises: bool = True
) -> bool:
    """Check that every tensor of a non-empty list is on the device of the first.

    ``KORNIA_CHECK_SAME_DEVICES(tensors)`` returns ``True`` when ``tensors`` is a non-empty ``list`` of tensors that are
    all on ``tensors[0].device``. A tuple, an empty list or an element that is not a tensor raises
    :exc:`~kornia.core.exceptions.TypeCheckError`, and tensors on different devices raise
    :exc:`~kornia.core.exceptions.DeviceError`.

    Args:
        tensors: a non-empty list of tensors.
        msg: message to show in the exception, or a function that returns it; see :func:`KORNIA_CHECK`.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        TypeCheckError: if tensors is not a non-empty list of tensors and raises is True.
        DeviceError: if the tensors are not all on the same device and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_SAME_DEVICE`: compares two tensors.
        - :func:`KORNIA_CHECK_IS_LIST_OF_TENSOR`: checks the list type only, and accepts an empty list.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(1, 3, 1)
        >>> KORNIA_CHECK_SAME_DEVICES([x1, x2], "Tensors not in the same device")
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if not (isinstance(tensors, list) and len(tensors) > 0 and all(isinstance(x, torch.Tensor) for x in tensors)):
        if raises:
            msg = _resolve_msg(msg)
            prefix = "Expected a non-empty list of tensors, got"
            suffix = "" if msg is None else f"\n  {msg}"
            if not isinstance(tensors, list):
                raise TypeCheckError(
                    f"{prefix} {type(tensors).__name__}.{suffix}", actual_type=type(tensors), expected_type=list
                )
            if len(tensors) == 0:
                raise TypeCheckError(f"{prefix} an empty list.{suffix}")
            for x in tensors:
                if not isinstance(x, torch.Tensor):
                    raise TypeCheckError(
                        f"{prefix} a list containing {type(x).__name__}.{suffix}",
                        actual_type=type(x),
                        expected_type=torch.Tensor,
                    )
        return False
    if not all(tensors[0].device == x.device for x in tensors):
        if raises:
            msg = _resolve_msg(msg)
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
    """Check that two tensors have exactly the same shape.

    ``KORNIA_CHECK_SAME_SHAPE(x, y)`` returns ``True`` when ``x.shape == y.shape``; shapes that only broadcast, such as
    ``(1, 3)`` and ``(2, 3)``, fail. On failure it raises :exc:`~kornia.core.exceptions.ShapeError`.

    Args:
        x: first tensor to evaluate.
        y: second tensor to evaluate.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        ShapeError: if the two tensors do not have the same shape and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_SAME_SIZE`: compares one dimension, optionally accepting a size of 1.
        - :func:`KORNIA_CHECK_SHAPE`: compares one tensor with a pattern instead of another tensor.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_SAME_SHAPE(x1, x2)
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
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


def KORNIA_CHECK_SAME_SIZE(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int = 0,
    allow_one: bool = False,
    msg: Optional[CheckMessage] = None,
    raises: bool = True,
) -> bool:
    """Check that two tensors have the same size along one dimension.

    ``KORNIA_CHECK_SAME_SIZE(x, y, dim)`` returns ``True`` when ``x.shape[dim] == y.shape[dim]``; with
    ``allow_one=True``, a size of 1 in ``x``, not in ``y``, also passes. A ``dim`` that either tensor lacks fails. On
    failure it raises :exc:`~kornia.core.exceptions.ShapeError`.

    Args:
        x: the tensor to evaluate.
        y: the reference tensor.
        dim: the dimension to compare; negative values count from the end.
        allow_one: also accept a size of 1 for ``x``, such as a kernel or a parameter shared by the whole batch.
        msg: message to append to the error, or a function that returns it; see :func:`KORNIA_CHECK`.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        ShapeError: if a tensor has no dimension ``dim``, or the sizes differ (and that of ``x`` is not 1 with
            ``allow_one``), and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_SAME_SHAPE`: compares every dimension, with no exception for a size of 1.
        - :func:`KORNIA_CHECK_SIZES`: compares every dimension with given sizes.

    Example:
        >>> image = torch.rand(4, 3, 8, 8)
        >>> sigma = torch.rand(1, 2)
        >>> KORNIA_CHECK_SAME_SIZE(sigma, image, allow_one=True)
        True
        >>> KORNIA_CHECK_SAME_SIZE(torch.rand(2, 2), image, raises=False)
        False

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not _KORNIA_CHECKS_ENABLED:
            return True

    # every ``.shape`` builds a new ``torch.Size``, so each is read once
    x_shape, y_shape = x.shape, y.shape
    x_ndim, y_ndim = len(x_shape), len(y_shape)
    in_range = -x_ndim <= dim < x_ndim and -y_ndim <= dim < y_ndim
    # A plain 1 passes before it is compared with the other size: under a size-oblivious torch.export, which the
    # Dynamo ONNX exporter of torch 2.9.1 runs, ``1 == s`` guards a dynamic ``s`` against 1, and an export whose
    # dynamic batch admits 1 then fails.
    if not in_range or not (
        (allow_one and isinstance(x_shape[dim], int) and x_shape[dim] == 1)
        or x_shape[dim] == y_shape[dim]
        or (allow_one and x_shape[dim] == 1)
    ):
        if raises:
            msg = _resolve_msg(msg)
            x_shape_list = list(x_shape)
            y_shape_list = list(y_shape)
            if not in_range:
                error_msg = f"Size mismatch: the tensors have no dimension {dim}.\n"
            else:
                expected = f"{y_shape[dim]} or 1" if allow_one and y_shape[dim] != 1 else f"{y_shape[dim]}"
                error_msg = f"Size mismatch at dimension {dim}: expected {expected}, got {x_shape[dim]}.\n"
            error_msg += f"  First tensor shape: {x_shape_list}\n"
            error_msg += f"  Second tensor shape: {y_shape_list}"
            if msg is not None:
                error_msg += f"\n  {msg}"
            raise ShapeError(error_msg, actual_shape=x_shape_list, expected_shape=y_shape_list)
        return False
    return True


def KORNIA_CHECK_SIZES(
    x: torch.Tensor, sizes: list[Optional[int]], msg: Optional[CheckMessage] = None, raises: bool = True
) -> bool:
    """Check a tensor's sizes against integers computed at run time, None for any size.

    ``KORNIA_CHECK_SIZES(x, [n, None])`` returns ``True`` when ``x`` has ``len(sizes)`` dimensions and every size that
    is not ``None`` equals the size of its dimension. On failure it raises :exc:`~kornia.core.exceptions.ShapeError`.
    Under a dynamic-shape ``torch.compile`` it compares the sizes without specializing them.

    Args:
        x: the tensor to evaluate.
        sizes: the expected size of every dimension, or ``None`` for a dimension of any size.
        msg: message to append to the error, or a function that returns it; see :func:`KORNIA_CHECK`.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        ShapeError: if the tensor does not have ``len(sizes)`` dimensions, or one of them differs from a given size, and
            raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_SHAPE`: takes a pattern of strings, with a ``"*"`` for any number of dimensions.
        - :func:`KORNIA_CHECK_SAME_SIZE`: compares one dimension with another tensor's, optionally accepting 1.

    Example:
        >>> pred = torch.rand(2, 5, 4, 4)
        >>> weight = torch.rand(5)
        >>> KORNIA_CHECK_SIZES(weight, [pred.shape[1]])
        True
        >>> KORNIA_CHECK_SIZES(pred, [2, None, 4, 4])
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not _KORNIA_CHECKS_ENABLED:
            return True

    x_shape = x.shape
    matches = len(x_shape) == len(sizes)
    if matches:
        for i in range(len(sizes)):
            size = sizes[i]
            if size is not None and x_shape[i] != size:
                matches = False
                break
    if not matches:
        if raises:
            msg = _resolve_msg(msg)
            expected: list[str] = []
            for size in sizes:
                expected.append("*" if size is None else str(size))
            x_shape_list = list(x.shape)
            expected_str = ", ".join(expected) + ("," if len(expected) == 1 else "")
            error_msg = f"Shape mismatch: expected ({expected_str}), got {x_shape_list}."
            if msg is not None:
                error_msg += f"\n  {msg}"
            raise ShapeError(error_msg, actual_shape=x_shape_list, expected_shape=expected)
        return False
    return True


def KORNIA_CHECK_IS_COLOR(x: torch.Tensor, msg: Optional[CheckMessage] = None, raises: bool = True) -> bool:
    """Check that an image tensor is color, with three channels at dimension -3.

    ``KORNIA_CHECK_IS_COLOR(x)`` returns ``True`` when ``x`` has at least three dimensions and ``x.shape[-3] == 3``, as
    in ``(*, 3, H, W)``. On failure it raises :exc:`~kornia.core.exceptions.ImageError`.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception, or a function that returns it; see :func:`KORNIA_CHECK`.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        ImageError: if ``x`` does not have a shape ``(*, 3, H, W)`` and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_IS_GRAY`: one channel, or a 2D ``(H, W)`` tensor.
        - :func:`KORNIA_CHECK_IS_COLOR_OR_GRAY`: one or three channels.
        - :func:`KORNIA_CHECK_IS_IMAGE`: one or three channels, and values in :math:`[0, 1]`, or up to ``2 ** bits - 1``
          for integers.

    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_IS_COLOR(img, "Image is not color")
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if len(x.shape) < 3 or x.shape[-3] != 3:
        if raises:
            msg = _resolve_msg(msg)
            error_msg = f"Not a color tensor. Got shape {list(x.shape)}."
            if msg is not None:
                error_msg += f"\n{msg}"
            raise ImageError(error_msg)
        return False
    return True


def KORNIA_CHECK_IS_GRAY(x: torch.Tensor, msg: Optional[CheckMessage] = None, raises: bool = True) -> bool:
    """Check that an image tensor is gray, with one channel at dimension -3 or none.

    ``KORNIA_CHECK_IS_GRAY(x)`` returns ``True`` for a ``(*, 1, H, W)`` tensor and for any 2D tensor, read as
    ``(H, W)``. On failure it raises :exc:`~kornia.core.exceptions.ImageError`.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception, or a function that returns it; see :func:`KORNIA_CHECK`.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        ImageError: if ``x`` is neither ``(*, 1, H, W)`` nor 2D and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_IS_COLOR`: three channels.
        - :func:`KORNIA_CHECK_IS_COLOR_OR_GRAY`: one or three channels, and rejects a 2D ``(H, W)`` tensor.

    Example:
        >>> img = torch.rand(2, 1, 4, 4)
        >>> KORNIA_CHECK_IS_GRAY(img, "Image is not grayscale")
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if len(x.shape) < 2 or (len(x.shape) >= 3 and x.shape[-3] != 1):
        if raises:
            msg = _resolve_msg(msg)
            error_msg = f"Not a gray tensor. Got shape {list(x.shape)}."
            if msg is not None:
                error_msg += f"\n{msg}"
            raise ImageError(error_msg)
        return False
    return True


def KORNIA_CHECK_IS_COLOR_OR_GRAY(x: torch.Tensor, msg: Optional[CheckMessage] = None, raises: bool = True) -> bool:
    """Check that an image tensor has one or three channels at dimension -3.

    ``KORNIA_CHECK_IS_COLOR_OR_GRAY(x)`` returns ``True`` when ``x`` is ``(*, 1, H, W)`` or ``(*, 3, H, W)``; unlike
    :func:`KORNIA_CHECK_IS_GRAY`, it rejects a 2D ``(H, W)`` tensor. It checks the shape only, not the dtype or the
    values. On failure it raises :exc:`~kornia.core.exceptions.ImageError`.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception, or a function that returns it; see :func:`KORNIA_CHECK`.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        ImageError: if ``x`` does not have a shape ``(*, 1, H, W)`` or ``(*, 3, H, W)`` and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_IS_IMAGE`: the same shape check, and values in :math:`[0, 1]`, or up to ``2 ** bits - 1``
          for integers.
        - :func:`KORNIA_CHECK_IS_COLOR`: three channels only.
        - :func:`KORNIA_CHECK_IS_GRAY`: one channel only, or a 2D ``(H, W)`` tensor.

    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_IS_COLOR_OR_GRAY(img, "Image is not color or grayscale")
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if len(x.shape) < 3 or x.shape[-3] not in [1, 3]:
        if raises:
            msg = _resolve_msg(msg)
            error_msg = f"Not a color or gray tensor. Got shape {list(x.shape)}."
            if msg is not None:
                error_msg += f"\n{msg}"
            raise ImageError(error_msg)
        return False
    return True


# The unsigned dtypes wider than 8 bits, each with the signed dtype of the same width.
_UNSIGNED_AS_SIGNED = {torch.uint16: torch.int16, torch.uint32: torch.int32, torch.uint64: torch.int64}


def KORNIA_CHECK_IS_IMAGE(
    x: torch.Tensor, msg: Optional[CheckMessage] = None, raises: bool = True, bits: int = 8
) -> bool:
    """Check that a tensor is a color or gray image in [0, 1], or [0, 255] for integers by default.

    ``KORNIA_CHECK_IS_IMAGE(x)`` returns ``True`` when ``x`` is ``(*, 1, H, W)`` or ``(*, 3, H, W)`` and its values lie
    in :math:`[0, 1]` for a floating dtype, or from ``0`` to ``2 ** bits - 1`` for an integer or boolean dtype of any
    width. A NaN or an infinity fails, and an empty image passes. The minimum and maximum are read on the host, so the
    check waits for an accelerator and breaks a ``torch.compile`` graph, and TorchScript cannot compile it.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception, or a function that returns it; see :func:`KORNIA_CHECK`.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.
        bits: the bit depth of an integer image; the default 8 accepts values up to 255. A floating image ignores it.

    Raises:
        ImageError: if the tensor shape is invalid and raises is True.
        ValueCheckError: if the tensor value range is invalid and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`KORNIA_CHECK_IS_COLOR_OR_GRAY`: the same shape check, without reading the values.

    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_IS_IMAGE(img, "It is not an image")
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if not KORNIA_CHECK_IS_COLOR_OR_GRAY(x, msg, raises):
        return False

    if x.numel() == 0:
        return True

    offset = 0
    if x.dtype in (torch.bfloat16, float16, float32, float64):
        low, high = 0.0, 1.0
    else:
        low, high = 0, (1 << bits) - 1
        signed = _UNSIGNED_AS_SIGNED.get(x.dtype)
        if signed is not None:
            # torch.aminmax has no kernel for these dtypes. Flipping the sign bit of a same-width signed view maps the
            # unsigned order onto the signed order; the offset is added back to the Python numbers below.
            x = x.view(signed) ^ torch.iinfo(signed).min
            offset = 1 << (torch.iinfo(signed).bits - 1)

    # The bounds are compared as Python numbers, so 2 ** bits - 1 cannot overflow the dtype, and in this form
    # a NaN, which fails every comparison, fails the check.
    amin, amax = torch.stack(torch.aminmax(x)).tolist()
    amin, amax = amin + offset, amax + offset
    if not (low <= amin and amax <= high):
        return _handle_invalid_range(msg, raises, amin, amax, (low, high))

    return True


def KORNIA_CHECK_DM_DESC(desc1: torch.Tensor, desc2: torch.Tensor, dm: torch.Tensor, raises: bool = True) -> bool:
    """Check that a distance matrix's first two sizes match the counts of two descriptor sets.

    ``KORNIA_CHECK_DM_DESC(desc1, desc2, dm)`` returns ``True`` when ``dm.shape[0] == desc1.shape[0]`` and
    ``dm.shape[1] == desc2.shape[0]``, as for ``(N, D)`` and ``(M, D)`` descriptors and their ``(N, M)`` distance
    matrix. Only these sizes are compared: further dimensions of ``dm`` and the descriptor size ``D`` are not checked,
    and a ``dm`` with fewer than two dimensions or a 0-dimensional descriptor fails. On failure it raises
    :exc:`~kornia.core.exceptions.ShapeError`.

    Args:
        desc1: first descriptor tensor to evaluate.
        desc2: second descriptor tensor to evaluate.
        dm: distance matrix tensor to evaluate.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        ShapeError: if the shapes of the descriptors do not match the distance matrix and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`disable_checks`.

    See Also:
        - :func:`~kornia.feature.match_nn`: a matcher that runs this check on a distance matrix passed as ``dm``.

    Example:
        >>> desc1 = torch.rand(4, 128)
        >>> desc2 = torch.rand(8, 128)
        >>> dm = torch.rand(4, 8)
        >>> KORNIA_CHECK_DM_DESC(desc1, desc2, dm)
        True

    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if dm.dim() < 2 or desc1.dim() < 1 or desc2.dim() < 1:
        if raises:
            raise ShapeError(
                "Distance matrix shape mismatch.\n"
                "  Expected a distance matrix with at least 2 dimensions and descriptors with at least 1.\n"
                f"  dm shape: {list(dm.shape)}, desc1 shape: {list(desc1.shape)}, desc2 shape: {list(desc2.shape)}",
                actual_shape=list(dm.shape),
            )
        return False
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
    """Check that a tensor has the shape of local affine frames, (B, N, 2, 3).

    ``KORNIA_CHECK_LAF(laf)`` returns ``True`` when ``laf`` is ``(B, N, 2, 3)``; it is
    ``KORNIA_CHECK_SHAPE(laf, ["B", "N", "2", "3"])``. On failure it raises :exc:`~kornia.core.exceptions.ShapeError`.

    Convention:
        Only the shape is checked: any dtype and any values pass, ``N = 0`` included.

    Args:
        laf: local affine frame tensor to evaluate.
        raises: whether to raise on failure; with ``False``, the check returns ``False`` instead.

    Raises:
        ShapeError: if ``laf`` does not have a shape ``(B, N, 2, 3)`` and raises is True.

    Note:
        Skipped while checks are disabled; see :func:`~kornia.core.check.disable_checks`.

    See Also:
        - :func:`~kornia.feature.laf_is_valid`: checks the values: finite entries and a nonzero determinant.
        - :func:`~kornia.core.check.KORNIA_CHECK_SHAPE`: checks any other shape pattern.

    Example:
        >>> lafs = torch.rand(2, 10, 2, 3)
        >>> KORNIA_CHECK_LAF(lafs)
        True

    """
    return KORNIA_CHECK_SHAPE(laf, ["B", "N", "2", "3"], raises=raises)


def _handle_invalid_range(
    msg: Optional[CheckMessage], raises: bool, min_val: float, max_val: float, expected_range: tuple[float, float]
) -> bool:
    """Handle invalid range cases."""
    if raises:
        low, high = (f"{b:g}" if isinstance(b, float) else f"{b}" for b in expected_range)
        err_msg = f"Value range mismatch: expected [{low}, {high}], got [{min_val}, {max_val}]."
        msg = _resolve_msg(msg)
        if msg is not None:
            err_msg += f"\n  {msg}"
        raise ValueCheckError(
            err_msg,
            actual_value=(min_val, max_val),
            expected_range=expected_range,
        )
    return False
