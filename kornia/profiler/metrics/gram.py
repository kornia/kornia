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
from torch import Tensor

from kornia.profiler.metrics.base import Metric


def gram_matrix(fmap: Tensor) -> Tensor:
    r"""Compute the normalized Gram matrix of a feature map.

    The feature map is flattened across its spatial dimensions, and the Gram
    matrix is computed as the batch-wise product of the flattened features
    and their transpose. The result is normalized by the number of elements
    in each channel's spatial feature map.

    Args:
        fmap: Input feature map with shape ``(B, C, H, W)``, where ``B`` is
            the batch size, ``C`` is the number of channels, and ``H`` and
            ``W`` are the spatial dimensions.

    Returns:
        A tensor of shape ``(B, C, C)`` containing the normalized Gram
        matrix for each sample in the batch.

    Raises:
        ValueError: If ``fmap`` is not a four-dimensional tensor.
    """
    if fmap.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got {fmap.shape}")

    batch_size, channels, height, width = fmap.shape
    fmap = fmap.reshape(batch_size, channels, -1)
    gram = torch.bmm(fmap, fmap.transpose(1, 2))

    return gram / (channels * height * width)


class GramSimilarity(Metric):
    r"""Compute Gram-based similarity between two feature-map tensors.

    Both inputs must have shape ``(B, C, H, W)``. Their normalized Gram
    matrices are flattened per sample, and cosine similarity is computed
    between the resulting representations. The final score is the mean
    cosine similarity across the batch.
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute the mean cosine similarity between two Gram matrices.

        Args:
            x: First feature-map tensor with shape ``(B, C, H, W)``.
            y: Second feature-map tensor with shape ``(B, C, H, W)``.

        Returns:
            A scalar tensor containing the mean cosine similarity across
            the batch.

        Raises:
            ValueError: If either input is not a four-dimensional tensor.
        """
        if x.dim() != 4 or y.dim() != 4:
            raise ValueError("GramSimilarity expects 4D tensors (B, C, H, W).")

        gram_x = gram_matrix(x)
        gram_y = gram_matrix(y)

        gram_x = gram_x.reshape(gram_x.size(0), -1)
        gram_y = gram_y.reshape(gram_y.size(0), -1)

        return F.cosine_similarity(gram_x, gram_y, dim=1).mean()
