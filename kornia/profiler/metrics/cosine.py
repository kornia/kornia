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


class CosineSimilarity(Metric):
    """Compute cosine similarity between two representation tensors.

    The input tensors are flattened across all dimensions except the batch
    dimension before their cosine similarity is computed.
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute the mean cosine similarity across the batch.

        Args:
            x: First representation tensor with shape ``(B, ...)``, where
                ``B`` is the batch size.
            y: Second representation tensor with shape ``(B, ...)``,
                matching ``x`` in batch size and flattened feature dimension.

        Returns:
            A scalar tensor containing the mean cosine similarity across
            the batch.
        """
        x = x.reshape(x.size(0), -1)
        y = y.reshape(y.size(0), -1)

        return F.cosine_similarity(x, y, dim=1).mean()
