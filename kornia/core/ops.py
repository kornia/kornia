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

import operator
from typing import Any

import torch

from kornia.core.check import KORNIA_CHECK, are_checks_enabled
from kornia.core.exceptions import TypeCheckError


def _check_n_is_integer(n: Any) -> None:
    """Raise `TypeCheckError` naming ``n`` unless it is an integer size.

    Accepts an ``int``, a NumPy integer, an integer tensor with one element and a ``SymInt``: whatever
    ``operator.index`` reads as an integer. ``bool`` is an ``int`` subclass and ``operator.index`` accepts it, as it
    does a boolean tensor, so both are rejected by name; ``torch.eye`` and ``torch.zeros`` refuse them as sizes
    anyway. An integer tensor that cannot be read (a meta tensor) raises torch's own ``RuntimeError``. Python-only:
    TorchScript cannot compile this body, and types ``n`` as ``int`` already. Like the ``KORNIA_CHECK*`` helpers, it
    does nothing once `disable_checks` has been called.
    """
    if not are_checks_enabled():
        return
    is_integer = True
    if isinstance(n, bool) or (isinstance(n, torch.Tensor) and n.dtype == torch.bool):
        is_integer = False
    elif not isinstance(n, (int, torch.SymInt)):
        # Ask `hasattr` first, so a `float`, `str` or `None` never reaches `operator.index`: on torch 2.5.1 dynamo
        # aborts the trace with `InternalTorchDynamoError` when a builtin raises inside a `try`.
        if not hasattr(n, "__index__"):
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
