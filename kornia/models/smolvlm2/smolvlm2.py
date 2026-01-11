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

from torch import nn
import torch

class SmolVLM2(nn.Module):
    """SmolVLM2 scaffold. This is a placeholder implementation."""
    def __init__(self, vision_dim: int = 768, text_dim: int = 768) -> None:
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, vision_dim)
        self.text_proj = nn.Linear(text_dim, text_dim)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        v = self.vision_proj(image_features)
        t = self.text_proj(text_features)
        return v + t
