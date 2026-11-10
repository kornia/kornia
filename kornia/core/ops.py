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

import math
import operator
from typing import Any, Dict, List, Tuple, Union

import torch

from kornia.core.check import KORNIA_CHECK, are_checks_enabled
from kornia.core.exceptions import TypeCheckError
from kornia.core.utils import is_compiling


def _is_numpy_bool(n: Any) -> bool:
    """Return whether ``n`` is a NumPy boolean scalar, read off its type so that this module does not import NumPy.

    NumPy 1.x boolean scalars implement ``__index__`` (deprecated there, removed in NumPy 2.0), so ``operator.index``
    reads ``np.True_`` as 1; the type is ``numpy.bool_`` on NumPy 1.x and ``numpy.bool`` on NumPy 2.
    """
    n_type = type(n)
    return n_type.__module__ == "numpy" and n_type.__name__ in ("bool_", "bool")


def _check_n_is_integer(n: Any) -> None:
    """Raise `TypeCheckError` naming ``n`` unless it is an integer size.

    Accepts an ``int``, a NumPy integer, an integer tensor with one element and a ``SymInt``: whatever
    ``operator.index`` reads as an integer. ``bool`` is an ``int`` subclass and ``operator.index`` accepts it, as it
    does a boolean tensor and, on NumPy 1.x, a NumPy boolean scalar, so all three are rejected by name; ``torch.eye``
    and ``torch.zeros`` refuse the first two as sizes anyway. An integer tensor that cannot be read (a meta tensor)
    raises torch's own ``RuntimeError``. Python-only: TorchScript cannot compile this body, and types ``n`` as ``int``
    already. Like the ``KORNIA_CHECK*`` helpers, it does nothing once `disable_checks` has been called.
    """
    if not are_checks_enabled():
        return
    is_integer = True
    if isinstance(n, bool) or (isinstance(n, torch.Tensor) and n.dtype == torch.bool):
        is_integer = False
    elif not isinstance(n, (int, torch.SymInt)):
        # Ask `hasattr` first, so a `float`, `str` or `None` never reaches `operator.index`: on torch 2.5.1 dynamo
        # aborts the trace with `InternalTorchDynamoError` when a builtin raises inside a `try`. The NumPy bool test
        # sits in this branch so that a plain `int` never meets it: torch 2.5.1 dynamo cannot trace its comparison.
        if not hasattr(n, "__index__") or _is_numpy_bool(n):
            is_integer = False
        else:
            try:
                operator.index(n)
            except TypeError:  # float tensor, multi-element tensor, ndarray; any other error (meta tensor) is torch's
                is_integer = False
    if not is_integer:
        raise TypeCheckError(
            f"n must be an integer. Got: {n!r} ({type(n).__name__})", actual_type=type(n), expected_type=int
        )


def _check_n(n: int) -> None:
    """Validate the ``n`` of `eye_like` and `vec_like`: an integer, else `TypeCheckError`, and a positive one."""
    if not torch.jit.is_scripting():
        _check_n_is_integer(n)
    KORNIA_CHECK(n > 0, f"n must be positive. Got: {n}")


def eye_like(n: int, input: torch.Tensor, shared_memory: bool = False) -> torch.Tensor:
    """Build a batch of ``N x N`` identity matrices with the batch size, dtype and device of a tensor.

    ``eye = eye_like(n, input)`` returns a ``(B, N, N)`` tensor holding one identity matrix per sample, where ``B`` is
    ``input.shape[0]``, with the dtype and device of ``input`` (a ``bool`` input gives ``True`` on the diagonal). The
    result does not require grad.

    Convention:
        - ``shared_memory=False`` (the default) returns a contiguous copy per sample, safe to write into.
        - ``shared_memory=True`` returns one identity expanded along the batch: writing into one sample, such as
          ``eye[0, 0, 0] = 5``, writes every sample, and for a batch of more than one, in-place arithmetic such as
          ``add_`` raises ``RuntimeError``.

    Args:
        n: the number of rows ``N``, a positive integer: an ``int``, a NumPy integer or a 0-d integer tensor.
            A ``float`` or a ``bool`` is rejected.
        input: the tensor whose batch size ``B``, dtype and device the result takes, of shape ``(B, *)``.
        shared_memory: whether to return an expanded view instead of a copy per sample.

    Returns:
        The identity matrices, ``(B, N, N)``.

    Raises:
        TypeCheckError: if ``n`` is not an integer.
        BaseError: if ``n`` is not positive, or if ``input`` has no dimension.

    See Also:
        - :func:`vec_like`: a batch of ``(N, 1)`` zero vectors instead.

    Example:
        >>> eye_like(2, torch.zeros(3, 5)).shape
        torch.Size([3, 2, 2])

    """
    _check_n(n)
    KORNIA_CHECK(len(input.shape) >= 1, f"input must have at least 1 dimension. Got shape: {input.shape}")

    # Use torch.eye with dtype parameter directly (available since PyTorch 2.0+)
    identity = torch.eye(n, device=input.device, dtype=input.dtype)

    return identity[None].expand(input.shape[0], n, n) if shared_memory else identity[None].repeat(input.shape[0], 1, 1)


def vec_like(n: int, tensor: torch.Tensor, shared_memory: bool = False) -> torch.Tensor:
    """Build a batch of ``N x 1`` zero column vectors with the batch size, dtype and device of a tensor.

    ``vec = vec_like(n, tensor)`` returns a ``(B, N, 1)`` tensor of zeros, where ``B`` is ``tensor.shape[0]``, with the
    dtype and device of ``tensor``. The result does not require grad.

    Convention:
        - ``shared_memory=False`` (the default) returns a contiguous copy per sample, safe to write into.
        - ``shared_memory=True`` returns one vector expanded along the batch: writing into one sample, such as
          ``vec[0, 0, 0] = 5``, writes every sample, and for a batch of more than one, in-place arithmetic such as
          ``add_`` raises ``RuntimeError``.

    Args:
        n: the number of rows ``N``, a positive integer: an ``int``, a NumPy integer or a 0-d integer tensor.
            A ``float`` or a ``bool`` is rejected.
        tensor: the tensor whose batch size ``B``, dtype and device the result takes, of shape ``(B, *)``.
        shared_memory: whether to return an expanded view instead of a copy per sample.

    Returns:
        The zero vectors, ``(B, N, 1)``.

    Raises:
        TypeCheckError: if ``n`` is not an integer.
        BaseError: if ``n`` is not positive, or if ``tensor`` has no dimension.

    See Also:
        - :func:`eye_like`: a batch of ``(N, N)`` identity matrices instead.

    Example:
        >>> vec_like(3, torch.zeros(2, 4)).shape
        torch.Size([2, 3, 1])

    """
    _check_n(n)
    KORNIA_CHECK(len(tensor.shape) >= 1, f"tensor must have at least 1 dimension. Got shape: {tensor.shape}")

    vec = torch.zeros(n, 1, device=tensor.device, dtype=tensor.dtype)
    return vec[None].expand(tensor.shape[0], n, 1) if shared_memory else vec[None].repeat(tensor.shape[0], 1, 1)


def _flatten_constant(data: Any, shape: List[int], leaves: List[Any], depth: int = 0) -> None:
    if isinstance(data, (list, tuple)):
        if depth == len(shape):
            shape.append(len(data))
        for value in data:
            _flatten_constant(value, shape, leaves, depth + 1)
    else:
        leaves.append(data)


def _constant_tensor(
    data: Union[float, List[Any], Tuple[Any, ...]],
    *,
    device: Union[str, torch.device, None] = None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct small numeric constants inside the graph, without lifting tensor storage.

    Inductor can reuse a lifted CPU tensor in a CUDA kernel without transferring it
    (https://github.com/pytorch/pytorch/issues/196969). Scalar factories and stacks
    avoid that path, including for constants returned alongside CUDA tensors. Indexed
    scalar writes are unsafe too: tracing lifts their right-hand sides.

    Eager execution, Dynamo and ``make_fx`` all run this same construction, which issues
    no host-device copy. Each distinct Python scalar is filled once and a single stack
    reuses it, so repeated coordinates such as box corners cost one kernel per value.

    ``data`` contains Python scalars or rectangular nested lists/tuples, never tensors.
    Other array-likes, such as NumPy arrays, keep ``torch.as_tensor`` semantics. Callers
    choose dtype explicitly and perform coordinate arithmetic before calling this helper
    when rounding before versus after casting matters.
    """
    if not isinstance(data, (list, tuple)):
        if isinstance(data, (int, float, torch.SymInt, torch.SymFloat)):
            return torch.full((), data, device=device, dtype=dtype)
        return torch.as_tensor(data, device=device, dtype=dtype)
    shape: List[int] = []
    leaves: List[Any] = []
    _flatten_constant(data, shape, leaves)
    if not leaves:
        return torch.empty(shape, device=device, dtype=dtype)
    filled: Dict[Tuple[type, Any], torch.Tensor] = {}
    values: List[torch.Tensor] = []
    for leaf in leaves:
        # Symbolic sizes stay unmerged because comparing them adds guards. The float key carries
        # the sign because -0.0 == 0.0; NaN is filled per leaf because NaN != NaN.
        if type(leaf) in (int, bool) or (type(leaf) is float and not math.isnan(leaf)):
            key = (type(leaf), leaf, math.copysign(1.0, leaf))
            if key not in filled:
                filled[key] = torch.full((), leaf, device=device, dtype=dtype)
            values.append(filled[key])
        else:
            values.append(torch.full((), leaf, device=device, dtype=dtype))
    return _stack_without_autocast(values).view(shape)


# The device types whose autocast state torch can report; ``torch.is_autocast_enabled("meta")`` raises.
_AUTOCAST_DEVICE_TYPES = ("cpu", "cuda", "mps", "xpu", "xla", "hpu")


def _stack_without_autocast(values: List[torch.Tensor]) -> torch.Tensor:
    """Stack ``values`` in their own dtype, also inside an autocast region.

    ``torch.stack`` is on autocast's promote list, which rejects a half dtype other than the region's own (a
    ``float16`` stack inside CPU autocast, whose lower precision is ``bfloat16``). A literal keeps the dtype it was
    asked for, as ``torch.tensor`` does, so autocast is switched off for the stack.
    """
    device_type = values[0].device.type
    if device_type in _AUTOCAST_DEVICE_TYPES and torch.is_autocast_enabled(device_type):
        with torch.autocast(device_type, enabled=False):
            return torch.stack(values)
    return torch.stack(values)


def _device_literal(
    data: Union[float, List[Any], Tuple[Any, ...]], *, device: Union[str, torch.device, None], dtype: torch.dtype
) -> torch.Tensor:
    """Return ``torch.tensor(data, device=device, dtype=dtype)`` for a literal of Python numbers, without a host sync.

    An eager call fills the literal on the device with :func:`_constant_tensor`: ``torch.tensor(..., device=cuda)``
    copies it from the host and waits for every queued kernel. Under ``torch.compile`` and ``torch.export`` the literal
    stays ``torch.tensor(..., device=device)``, as before: capture makes it a constant of the graph on ``device``, which
    costs no copy at run time and which Inductor folds into its consumer, where filled values would add kernels. Both
    round the same Python numbers to ``dtype``, so the values are the same.
    """
    if is_compiling() or (dtype == torch.float16 and not _fits_float16(data)):
        return torch.tensor(data, device=device, dtype=dtype)
    return _constant_tensor(data, device=device, dtype=dtype)


def _fits_float16(data: Union[float, List[Any], Tuple[Any, ...]]) -> bool:
    """Whether every finite number in ``data`` is within the float16 range.

    ``torch.full`` raises for a finite value beyond it on CUDA (and on the CPU in torch 2.5.1), where ``torch.tensor``
    rounds it to 65504 or to infinity, so such a literal keeps ``torch.tensor``.
    """
    leaves: List[Any] = []
    _flatten_constant(data, [], leaves)
    return all(not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) <= 65504.0 for v in leaves)
