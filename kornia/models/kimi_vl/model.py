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

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .config import KimiVLConfig, KimiVLProjectorConfig
from .moonvit import MoonViT

__all__ = ["KimiVLModel", "KimiVLProjector"]


class KimiVLProjector(nn.Module):
    """KimiVL Projector with Pixel Unshuffle and MLP."""

    def __init__(self, config: KimiVLProjectorConfig) -> None:
        super().__init__()
        self.downsample_ratio = 2

        # Pre-norm (applied before pixel unshuffle, on the vision encoder output dimension)
        self.pre_norm = nn.LayerNorm(config.input_dim)

        # After pixel unshuffle, the dimension becomes input_dim * (downsample_ratio ** 2)
        mlp_input_dim = config.input_dim * (self.downsample_ratio**2)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.Dropout(config.dropout_p),
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, N, D)
            h: Height in patches
            w: Width in patches

        Returns:
            Projected features (B, N/4, output_dim)
        """
        B, _N, D = x.shape

        # Apply pre-norm
        x = self.pre_norm(x)

        # Reshape to spatial (B, H, W, D) -> (B, D, H, W)
        x = x.view(B, h, w, D).permute(0, 3, 1, 2)

        # Pixel unshuffle for spatial downsampling -> (B, D*4, H/2, W/2)
        x = torch.nn.functional.pixel_unshuffle(x, self.downsample_ratio)

        # Flatten back -> (B, D*4, N/4) -> (B, N/4, D*4)
        x = x.flatten(2).transpose(1, 2)

        # MLP
        x = self.mlp(x)

        return x


class KimiVLModel(nn.Module):
    """KimiVL Vision-Language Model (Vision Part).

    This model includes the Vision Encoder (MoonViT) and the Projector.
    It does not include the LLM decoder, as Kornia focuses on the vision components.

    Args:
        config: KimiVL configuration.
    """

    def __init__(self, config: KimiVLConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.vision_config.patch_size

        # Vision Encoder (MoonViT)
        self.vision_encoder = MoonViT(config.vision_config)

        # Projector
        self.projector = KimiVLProjector(config.projector_config)

    def forward(self, images: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            images: (B, C, H, W)
            attention_mask: (B, 1, N, N) or (B, N, N) optional mask for vision encoder.

        Returns:
            Projected visual features (B, seq_len, output_dim)
        """
        vision_features = self.vision_encoder(images, attention_mask=attention_mask)  # (B, N, D)

        # Calculate patch grid size
        H, W = images.shape[2], images.shape[3]
        patch_size = self.patch_size
        h_patches = H // patch_size
        w_patches = W // patch_size
        projected_features = self.projector(vision_features, h_patches, w_patches)

        return projected_features
