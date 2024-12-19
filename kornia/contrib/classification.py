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

"""Module containing utilities for classification."""

import torch
from torch import nn


class ClassificationHead(nn.Module):
    """Module to be used as a classification head.

    Args:
        embed_size: the logits tensor coming from the networks.
        num_classes: an integer representing the numbers of classes to classify.

    Example:
        >>> feat = torch.rand(1, 256, 256)
        >>> head = ClassificationHead(256, 10)
        >>> head(feat).shape
        torch.Size([1, 10])

    """

    def __init__(self, embed_size: int = 768, num_classes: int = 10) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.mean(-2)
        return self.linear(self.norm(out))
