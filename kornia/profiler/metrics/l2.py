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


class L2Normalized(Metric):
    r"""Compute the normalized L2 distance between two representation tensors.

    The L2 distance is computed independently for each sample and normalized
    by the L2 norm of the reference tensor. The normalized distances are then
    averaged across the batch.

    The normalization includes a small constant to avoid division by zero
    when the reference tensor has a zero norm.
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Compute the mean normalized L2 distance.

        Args:
            x: Reference tensor with shape ``(B, ...)``, where ``B`` is the
                batch size.
            y: Comparison tensor with shape ``(B, ...)``, with the same batch
                size and flattened feature dimension as ``x``.

        Returns:
            A scalar tensor containing the mean normalized L2 distance
            across the batch.
        """
        x = x.reshape(x.size(0), -1)
        y = y.reshape(y.size(0), -1)

        l2 = torch.norm(x - y, p=2, dim=1)
        norm = torch.norm(x, p=2, dim=1)

        normalized = l2 / (norm + 1e-8)
        return normalized.mean()
