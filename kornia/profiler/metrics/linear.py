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


def linear_cka(x: Tensor, y: Tensor) -> float:
    r"""Compute biased linear CKA across corresponding samples.

    Each column is centered across samples. The score is the normalized
    Frobenius inner product of the sample Gram matrices. Feature dimensions
    may differ, but sample counts and devices must match.

    Args:
        x: Floating-point feature matrix of shape :math:`(N, D_x)`.
        y: Floating-point feature matrix of shape :math:`(N, D_y)`.

    Returns:
        Scalar similarity score. Computation uses float64 for stability.

    Raises:
        ValueError: For invalid shapes, devices, nonfinite inputs, or zero
            centered Gram matrices, for which the score is undefined.
        TypeError: If inputs are not real floating-point tensors.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("CKA requires 2D matrices with at least two corresponding samples")
    if x.shape[1] == 0 or y.shape[1] == 0 or x.device != y.device:
        raise ValueError("CKA requires nonempty feature dimensions on the same device")
    if not x.is_floating_point() or not y.is_floating_point():
        raise TypeError("CKA requires real floating-point inputs")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("CKA requires finite inputs")
    x, y = x.to(torch.float64), y.to(torch.float64)
    x, y = x - x.mean(dim=0, keepdim=True), y - y.mean(dim=0, keepdim=True)
    gram_x, gram_y = x @ x.T, y @ y.T
    denominator = torch.linalg.vector_norm(gram_x) * torch.linalg.vector_norm(gram_y)
    if not torch.isfinite(denominator) or denominator == 0:
        raise ValueError("CKA undefined for zero or nonfinite centered Gram norms")
    return ((gram_x * gram_y).sum() / denominator).item()


def linear_similarity(f1: Dict[str, Tensor], f2: Dict[str, Tensor]) -> Dict[str, float]:
    """Compute linear CKA for every matching layer, flattening features per sample."""
    if f1.keys() != f2.keys():
        raise ValueError("Feature dictionaries must have same layers")
    return {layer: linear_cka(f1[layer].flatten(1), f2[layer].flatten(1)) for layer in f1}
