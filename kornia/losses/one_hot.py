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

import torch
import torch.nn.functional as F

from kornia.core._compat import torch_version_ge
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR
from kornia.core.utils import is_compiling, is_exporting

# Inductor lowers torch._assert_async to a runtime check from torch 2.9.0; up to 2.8 it decomposes it to nothing.
_INDUCTOR_KEEPS_ASSERT_ASYNC = torch_version_ge(2, 9)


def one_hot(
    labels: torch.Tensor, num_classes: int, device: torch.device, dtype: torch.dtype, eps: float = 0.0
) -> torch.Tensor:
    r"""Convert an integer label x-D torch.Tensor to a one-hot (x+1)-D torch.Tensor.

    Convention:
        - int64 ``labels`` of shape ``(N, *)`` become ``(N, C, *)``: the class axis is dim 1, where
          :func:`~torch.nn.functional.one_hot` puts it last. With the default ``eps=0.0`` the result is
          ``F.one_hot(labels, num_classes).movedim(-1, 1).to(device, dtype)``. Labels outside ``[0, num_classes)``
          are not validated by kornia; the result for them is undefined.

    Args:
        labels: torch.Tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned torch.Tensor.
        dtype: the desired data type of returned torch.Tensor.
        eps: optional smoothing floor. The default 0 returns the exact one-hot. A non-zero value replaces the
          zeros with ``eps`` and keeps the ones, which needs a floating point ``dtype``.

    Returns:
        the labels in one hot torch.Tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3, device=torch.device('cpu'), dtype=torch.float32)
        tensor([[[[1., 0.],
                  [0., 1.]],
        <BLANKLINE>
                 [[0., 1.],
                  [0., 0.]],
        <BLANKLINE>
                 [[0., 0.],
                  [1., 0.]]]])

    """
    KORNIA_CHECK_IS_TENSOR(labels, "Input labels must be a torch.Tensor")
    KORNIA_CHECK(labels.dtype == torch.int64, f"labels must be of dtype torch.int64. Got: {labels.dtype}")
    KORNIA_CHECK(num_classes >= 1, f"The number of classes must be >= 1. Got: {num_classes}")
    if eps != 0:
        # TorchScript represents a ``torch.dtype`` as an int, which has no ``is_floating_point``.
        if torch.jit.is_scripting():
            is_floating = torch.empty(0, dtype=dtype).is_floating_point()
        else:
            is_floating = dtype.is_floating_point
        KORNIA_CHECK(is_floating, f"A non-zero eps needs a floating point dtype. Got: {dtype}")

    # Use PyTorch's built-in one_hot function
    one_hot_tensor = F.one_hot(labels, num_classes=num_classes)

    # PyTorch's one_hot adds the class dimension at the end: (*, num_classes)
    # We need it at position 1: (N, C, *)
    # Permute: move the last dimension (num_classes) to position 1
    ndim = labels.ndim
    permute_dims = [0] + [ndim] + list(range(1, ndim))
    one_hot_tensor = one_hot_tensor.permute(permute_dims)

    one_hot_tensor = one_hot_tensor.to(dtype=dtype, device=device)
    if eps == 0:
        return one_hot_tensor

    # Optional smoothing floor: the zeros become eps and the ones stay 1
    return one_hot_tensor * (1.0 - eps) + eps


def _check_class_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Raise :func:`one_hot`'s errors for CPU labels outside ``[0, num_classes)``, eager and compiled alike.

    For a loss that gathers, scatters or indexes with the labels instead of building a one-hot target: index with the
    labels it returns. Inductor's CPU kernel for such an op checks each index inside an OpenMP parallel region, where a
    failed check cannot propagate, so an out-of-range label terminates the process (``index out of bounds``, exit code
    134) instead of raising the eager op's ``RuntimeError``. ``torch._assert_async`` raises a ``RuntimeError`` on the
    CPU, at once in eager code and from the compiled kernel that evaluates it, without a graph break.

    Inductor may fuse the asserts into the index op's kernel and evaluate them after its parallel loop, so under
    ``torch.compile`` the labels come back clamped into range: the index op stays in bounds until the asserts raise.
    They stay unclamped where the asserts may not survive, so that the index op fails instead of a wrong loss being
    returned: before torch 2.9, whose Inductor compiles the assert to nothing, and in an exported graph, which ONNX
    translates without it. torch 2.9.1 reports a ``torch.compile`` capture as an export too; its Inductor ran the
    asserts first in every case checked.

    Only CPU labels are checked: on CUDA the index kernels fail an out-of-range label with a device-side assert in
    eager and compiled code alike, and so would this assert. Functorch transforms skip it, since ``vmap`` has no
    batching rule for the assert, and so does TorchScript, whose scripted and traced graphs drop the assert anyway;
    the index op raises its own ``RuntimeError`` there.

    Args:
        labels: int64 class labels of any shape; labels to ignore must already be replaced by a valid class.
        num_classes: the number of classes.

    Returns:
        ``labels``, clamped to ``[0, num_classes - 1]`` in a CPU ``torch.compile`` capture that is not an export,
        from torch 2.9 on.
    """
    if labels.device.type != "cpu":
        return labels
    if not torch.jit.is_scripting():  # TorchScript only skips a bare is_scripting() guard
        if torch._C._are_functorch_transforms_active():
            return labels
        # One pass for both bounds (a quarter of the eager cost of two comparisons and two all()); aminmax rejects an
        # empty tensor, which holds no label to check. An explicit dimension lets ONNX lower the reduction too.
        if labels.numel() > 0:
            low, high = torch.aminmax(labels.reshape(-1), dim=0)
            torch._assert_async(low >= 0, "Class values must be non-negative.")
            torch._assert_async(high < num_classes, "Class values must be smaller than num_classes.")
        if _INDUCTOR_KEEPS_ASSERT_ASYNC and is_compiling() and not is_exporting():
            return labels.clamp(0, num_classes - 1)
    return labels
