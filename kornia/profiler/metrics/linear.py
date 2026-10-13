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

import torch.nn.functional as F
from torch import Tensor

from kornia.profiler.metrics.base import Metric


def linear_cka(x: Tensor, y: Tensor) -> Tensor:
    r"""Compute the linear similarity between two representation tensors.

    Each input is centered along its feature dimension and L2-normalized
    before the element-wise product is summed across features. The resulting
    similarity scores are averaged across the batch.

    Args:
        x: First representation tensor of shape ``(B, D)``, where ``B`` is
            the batch size and ``D`` is the feature dimension.
        y: Second representation tensor of shape ``(B, D)``, matching ``x``
            in shape.

    Returns:
        A scalar tensor containing the mean similarity across the batch.
    """
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)

    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)

    return (x * y).sum(dim=1).mean()


class LinearSimilarity(Metric):
    r"""Compute linear similarity between two representation tensors.

    Inputs are flattened across all dimensions except the batch dimension
    before the similarity calculation is performed.
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Compute the mean linear similarity across the batch.

        Args:
            x: First representation tensor with shape ``(B, ...)``, where
                ``B`` is the batch size.
            y: Second representation tensor with shape ``(B, ...)``, matching
                ``x`` in batch size and flattened feature dimension.

        Returns:
            A scalar tensor containing the mean similarity across the batch.
        """
        x = x.reshape(x.size(0), -1)
        y = y.reshape(y.size(0), -1)

        return linear_cka(x, y)
