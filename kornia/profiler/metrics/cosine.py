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

from typing import Dict

import torch
import torch.nn.functional as F


def cosine_similarity(
    f1: Dict[str, torch.Tensor],
    f2: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute layer-wise cosine similarity between feature dictionaries.

    Each dictionary maps layer names to feature tensors. The features are
    flattened and cosine similarity is computed independently for each layer.

    Args:
        f1: Dictionary mapping layer names to tensors of shape (B, ...).
        f2: Dictionary mapping layer names to tensors of shape (B, ...).

    Returns:
        Dictionary mapping layer names to mean cosine similarity across batch.
    """
    if f1.keys() != f2.keys():
        raise ValueError("Feature dictionaries must have same layers")

    results = {}

    for layer, x in f1.items():
        x = x.reshape(x.size(0), -1)
        y = f2[layer].reshape(f2[layer].size(0), -1)

        sim = F.cosine_similarity(x, y, dim=1).mean().item()
        results[layer] = sim

    return results
