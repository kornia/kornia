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
    r"""Return a 2-D tensor with ones on the diagonal and zeros elsewhere with the same batch size as the input.

    Args:
        n: the number of rows :math:`(N)`, a positive integer: an ``int``, a NumPy integer or a 0-d integer tensor.
          A ``float`` or a ``bool`` is rejected.
        input: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.
        shared_memory: when set, all samples in the batch will share the same memory.

    Returns:
       The identity matrix with the same batch size as the input :math:`(B, N, N)`.

    Raises:
        TypeCheckError: if ``n`` is not an integer.
        BaseError: if ``n`` is not positive, or if ``input`` has no dimension.

    Notes:
        When the dimension to expand is of size 1, using torch.expand(...) yields the same tensor as torch.repeat(...)
        without using extra memory. Thus, when the tensor obtained by this method will be later assigned -
        use this method with shared_memory=False, otherwise, prefer using it with shared_memory=True.

    """
    _check_n(n)
    KORNIA_CHECK(len(input.shape) >= 1, f"input must have at least 1 dimension. Got shape: {input.shape}")

    # Use torch.eye with dtype parameter directly (available since PyTorch 2.0+)
    identity = torch.eye(n, device=input.device, dtype=input.dtype)

    return identity[None].expand(input.shape[0], n, n) if shared_memory else identity[None].repeat(input.shape[0], 1, 1)


def vec_like(n: int, tensor: torch.Tensor, shared_memory: bool = False) -> torch.Tensor:
    r"""Return a 2-D tensor with a vector containing zeros with the same batch size as the input.

    Args:
        n: the number of rows :math:`(N)`, a positive integer: an ``int``, a NumPy integer or a 0-d integer tensor.
          A ``float`` or a ``bool`` is rejected.
        tensor: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.
        shared_memory: when set, all samples in the batch will share the same memory.

    Returns:
        The vector with the same batch size as the input :math:`(B, N, 1)`.

    Raises:
        TypeCheckError: if ``n`` is not an integer.
        BaseError: if ``n`` is not positive, or if ``tensor`` has no dimension.

    Notes:
        When the dimension to expand is of size 1, using torch.expand(...) yields the same tensor as torch.repeat(...)
        without using extra memory. Thus, when the tensor obtained by this method will be later assigned -
        use this method with shared_memory=False, otherwise, prefer using it with shared_memory=True.

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
    return torch.stack(values).view(shape)
