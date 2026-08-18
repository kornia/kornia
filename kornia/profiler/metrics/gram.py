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

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def gram_matrix(fmap: Tensor) -> Tensor:
    r"""Compute the Gram matrix for a feature map.

    The Gram matrix captures correlations between feature channels and is
    commonly used in style-based similarity metrics.

    Args:
        fmap: Input feature map of shape :math:`(B, C, H, W)`.

    Returns:
        Gram matrix of shape :math:`(B, C, C)`.

    Raises:
        ValueError: If the input tensor is not 4-dimensional.
    """
    if fmap.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got {fmap.shape}")

    B, C, H, W = fmap.shape
    fmap = fmap.reshape(B, C, -1)
    G = torch.bmm(fmap, fmap.transpose(1, 2))
    return G / (C * H * W)


def gram_similarity(
    f1: Dict[str, Tensor],
    f2: Dict[str, Tensor],
) -> Dict[str, Optional[float]]:
    r"""Compute Gram-based similarity between two feature dictionaries.

    For each matching layer, computes the cosine similarity between the
    flattened Gram matrices of the corresponding feature maps.

    Only 4D feature maps are considered. Non-4D entries will return ``None``.

    Args:
        f1: Dictionary mapping layer names to feature tensors.
        f2: Dictionary mapping layer names to feature tensors.

    Returns:
        Dictionary mapping layer names to similarity scores (float). If a layer
        does not contain valid 4D tensors, the value is ``None``.

    Raises:
        ValueError: If the dictionaries do not have identical keys.
    """
    if f1.keys() != f2.keys():
        raise ValueError("Feature dictionaries must have same layers")

    results: Dict[str, Optional[float]] = {}

    for layer, x in f1.items():
        y = f2[layer]

        if x.dim() != 4 or y.dim() != 4:
            results[layer] = None
            continue

        G1 = gram_matrix(x)
        G2 = gram_matrix(y)

        G1_flat = G1.reshape(G1.size(0), -1)
        G2_flat = G2.reshape(G2.size(0), -1)

        sim = F.cosine_similarity(G1_flat, G2_flat, dim=1).mean().item()
        results[layer] = sim

        return results
