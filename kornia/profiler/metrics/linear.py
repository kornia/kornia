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
from torch import Tensor

from kornia.profiler.metrics.base import Metric


def linear_cka(x: Tensor, y: Tensor) -> Tensor:
    r"""Compute linear Centered Kernel Alignment (CKA).

    Each representation is centered across the batch dimension. Linear CKA
    is computed as the squared Frobenius norm of the cross-covariance-like
    product, normalized by the Frobenius norms of the two self-products.

    Args:
        x: First representation tensor of shape ``(B, D)``, where ``B`` is
            the batch size and ``D`` is the feature dimension.
        y: Second representation tensor of shape ``(B, D)``, matching ``x``
            in shape.

    Returns:
        A scalar tensor containing the linear CKA similarity.
    """
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    xy = x @ y.T
    xx = x @ x.T
    yy = y @ y.T

    numerator = xy.square().sum()
    denominator = torch.linalg.norm(xx) * torch.linalg.norm(yy)

    eps = torch.finfo(denominator.dtype).eps
    return torch.where(denominator > 0, numerator / denominator.clamp_min(eps), 0.0)


class LinearSimilarity(Metric):
    r"""Compute linear CKA between two representation tensors.

    Inputs are flattened across all dimensions except the batch dimension
    before the CKA calculation is performed.
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Compute linear CKA between two representation tensors.

        Args:
            x: First representation tensor with shape ``(B, ...)``, where
                ``B`` is the batch size.
            y: Second representation tensor with shape ``(B, ...)``, matching
                ``x`` in batch size and flattened feature dimension.

        Returns:
            A scalar tensor containing the linear CKA similarity.
        """
        x = x.reshape(x.size(0), -1)
        y = y.reshape(y.size(0), -1)

        return linear_cka(x, y)
