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

"""NormalizeWithGrad — the carved-out gradient-preserving Normalize.

Use this when gradients need to flow through normalization. Standard
``kornia.augmentations.Normalize`` runs under ``@torch.no_grad()``; this
class explicitly removes that restriction for in-model preprocessing.

Math is identical to ``kornia.augmentation.Normalize``: ``(x - mean) / std``,
broadcast over the channel dimension.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class NormalizeWithGrad(nn.Module):
    """Channel-wise normalization that preserves gradients.

    Args:
        mean: Per-channel mean values. Length must equal C.
        std: Per-channel standard deviations. Length must equal C; all > 0.

    Example:
        >>> import torch
        >>> from kornia.augmentations.deterministic import NormalizeWithGrad
        >>> m = NormalizeWithGrad(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        >>> x = torch.ones(1, 3, 4, 4, requires_grad=True)
        >>> y = m(x)
        >>> y.sum().backward()
        >>> bool(x.grad is not None)
        True
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        std_t = torch.as_tensor(std, dtype=torch.float32)
        if mean_t.numel() != std_t.numel():
            raise ValueError(f"mean and std must have same length; got {mean_t.numel()} vs {std_t.numel()}")
        if (std_t <= 0).any():
            raise ValueError(f"std must be all > 0; got {std_t.tolist()}")
        # Register as buffers so they move with .to(device)
        self.register_buffer("mean", mean_t.view(1, -1, 1, 1))
        self.register_buffer("std", std_t.view(1, -1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() not in (3, 4):
            raise ValueError(f"input must be 3D or 4D; got {x.dim()}D")
        if x.dim() == 3:
            x = x.unsqueeze(0)
            return ((x - self.mean.to(x.dtype)) / self.std.to(x.dtype)).squeeze(0)
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    def extra_repr(self) -> str:
        return f"mean={self.mean.flatten().tolist()}, std={self.std.flatten().tolist()}"
