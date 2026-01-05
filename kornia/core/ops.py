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

from kornia.core.check import KORNIA_CHECK


def eye_like(n: int, input: torch.Tensor, shared_memory: bool = False) -> torch.Tensor:
    r"""Return a 2-D tensor with ones on the diagonal and zeros elsewhere with the same batch size as the input.

    Args:
        n: the number of rows :math:`(N)`.
        input: image tensor that will determine the batch size of the output matrix.
            The expected shape is :math:`(B, *)`.
        shared_memory: when set, all samples in the batch will share the same memory.

    Returns:
       The identity matrix with the same batch size as the input :math:`(B, N, N)`.

    Notes:
        When the dimension to expand is of size 1, using torch.expand(...) yields the same tensor as torch.repeat(...)
        without using extra memory. Thus, when the tensor obtained by this method will be later assigned -
        use this method with shared_memory=False, otherwise, prefer using it with shared_memory=True.

    """
    KORNIA_CHECK(n > 0, f"n must be positive. Got: {n}")
    KORNIA_CHECK(len(input.shape) >= 1, f"input must have at least 1 dimension. Got shape: {input.shape}")

    # Use torch.eye with dtype parameter directly (available since PyTorch 2.0+)
    identity = torch.eye(n, device=input.device, dtype=input.dtype)

    return identity[None].expand(input.shape[0], n, n) if shared_memory else identity[None].repeat(input.shape[0], 1, 1)


def vec_like(n: int, tensor: torch.Tensor, shared_memory: bool = False) -> torch.Tensor:
    r"""Return a 2-D tensor with a vector containing zeros with the same batch size as the input.

    Args:
        n: the number of rows :math:`(N)`.
        tensor: image tensor that will determine the batch size of the output matrix.
            The expected shape is :math:`(B, *)`.
        shared_memory: when set, all samples in the batch will share the same memory.

    Returns:
        The vector with the same batch size as the input :math:`(B, N, 1)`.

    Notes:
        When the dimension to expand is of size 1, using torch.expand(...) yields the same tensor as torch.repeat(...)
        without using extra memory. Thus, when the tensor obtained by this method will be later assigned -
        use this method with shared_memory=False, otherwise, prefer using it with shared_memory=True.

    """
    KORNIA_CHECK(n > 0, f"n must be positive. Got: {n}")
    KORNIA_CHECK(len(tensor.shape) >= 1, f"tensor must have at least 1 dimension. Got shape: {tensor.shape}")

    vec = torch.zeros(n, 1, device=tensor.device, dtype=tensor.dtype)
    return vec[None].expand(tensor.shape[0], n, 1) if shared_memory else vec[None].repeat(tensor.shape[0], 1, 1)
