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
from torch import Tensor


def l2_normalized(
    f1: Dict[str, Tensor],
    f2: Dict[str, Tensor],
) -> Dict[str, float]:
    r"""Compute normalized L2 distance between feature dictionaries.

    For each matching layer, computes the L2 distance between flattened
    feature tensors and normalizes it by the L2 norm of the reference tensor.

    The normalization is performed per sample and then averaged across the batch.

    Args:
        f1: Dictionary mapping layer names to feature tensors.
        f2: Dictionary mapping layer names to feature tensors.

    Returns:
        Dictionary mapping layer names to normalized L2 distance (float).

    Raises:
        ValueError: If the dictionaries do not have identical keys.
    """
    if f1.keys() != f2.keys():
        raise ValueError("Feature dictionaries must have same layers")

    results: Dict[str, float] = {}

    for layer, x in f1.items():
        x = x.reshape(x.size(0), -1)
        y = f2[layer].reshape(f2[layer].size(0), -1)

        l2 = torch.norm(x - y, p=2, dim=1)
        norm = torch.norm(x, p=2, dim=1)

        normalized = l2 / (norm + 1e-8)
        results[layer] = normalized.mean().item()

    return results
