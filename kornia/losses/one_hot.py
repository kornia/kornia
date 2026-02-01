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

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR


def one_hot(
    labels: torch.Tensor, num_classes: int, device: torch.device, dtype: torch.dtype, eps: float = 1e-6
) -> torch.Tensor:
    r"""Convert an integer label x-D torch.Tensor to a one-hot (x+1)-D torch.Tensor.

    Args:
        labels: torch.Tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned torch.Tensor.
        dtype: the desired data type of returned torch.Tensor.
        eps: epsilon for numerical stability.

    Returns:
        the labels in one hot torch.Tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3, device=torch.device('cpu'), dtype=torch.float32)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    KORNIA_CHECK_IS_TENSOR(labels, "Input labels must be a torch.Tensor")
    KORNIA_CHECK(labels.dtype == torch.int64, f"labels must be of dtype torch.int64. Got: {labels.dtype}")
    KORNIA_CHECK(num_classes >= 1, f"The number of classes must be >= 1. Got: {num_classes}")

    # Use PyTorch's built-in one_hot function
    one_hot_tensor = F.one_hot(labels, num_classes=num_classes)

    # PyTorch's one_hot adds the class dimension at the end: (*, num_classes)
    # We need it at position 1: (N, C, *)
    # Permute: move the last dimension (num_classes) to position 1
    ndim = labels.ndim
    permute_dims = [0] + [ndim] + list(range(1, ndim))
    one_hot_tensor = one_hot_tensor.permute(*permute_dims)

    # Convert to desired dtype and device, then apply eps for numerical stability
    one_hot_tensor = one_hot_tensor.to(dtype=dtype, device=device)
    # Apply eps: multiply by (1-eps) and add eps to all elements
    one_hot_tensor = one_hot_tensor * (1.0 - eps) + eps

    return one_hot_tensor
