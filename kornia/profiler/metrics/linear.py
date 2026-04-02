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


def linear_cka(x, y):
    r"""Compute linear CKA similarity between two tensors.

    The inputs are first centered along the feature dimension and then
    L2-normalized. The similarity is computed as the mean cosine similarity
    between corresponding samples.

    Args:
        x: Input tensor of shape :math:`(B, D)`.
        y: Input tensor of shape :math:`(B, D)`.

    Returns:
        Scalar similarity score (float), averaged over the batch.
    """
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)

    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)

    return (x * y).sum(dim=1).mean().item()


def linear_similarity(f1, f2):
    r"""Apply linear CKA similarity layer-wise between feature dictionaries.

    For each matching layer, feature tensors are flattened into 2D tensors
    of shape :math:`(B, D)` and compared using ``linear_cka``.

    Args:
        f1: Dictionary mapping layer names to feature tensors.
        f2: Dictionary mapping layer names to feature tensors.

    Returns:
        Dictionary mapping each layer name to its similarity score (float).

    Raises:
        ValueError: If the input dictionaries do not have identical keys.
    """
    if f1.keys() != f2.keys():
        raise ValueError("Feature dictionaries must have same layers")

    results = {}

    for layer in f1:
        x = f1[layer].reshape(f1[layer].size(0), -1)
        y = f2[layer].reshape(f2[layer].size(0), -1)

        results[layer] = linear_cka(x, y)

    return results
