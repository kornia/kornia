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

import math
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["NaFlex"]


class NaFlex(nn.Module):
    """NaFlex wrapper for vision embeddings that supports flexible image sizes.

    This module wraps a patch embedding function and a position embedding tensor.
    It computes patch embeddings and dynamically interpolates the position embeddings
    to match the input resolution using bilinear interpolation.

    Args:
        patch_embedding_fcn: A callable (e.g. Conv2d) that takes images and returns
            patch embeddings.
        position_embedding: The position embedding tensor to be interpolated.
    """

    def __init__(
        self,
        patch_embedding_fcn: Callable[[torch.Tensor], torch.Tensor],
        position_embedding: torch.Tensor,
    ) -> None:
        super().__init__()
        self.patch_embedding_fcn = patch_embedding_fcn
        self.position_embedding = position_embedding

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 1. Run the patch embedding function
        embeddings = self.patch_embedding_fcn(pixel_values)

        # 2. Infer grid dimensions
        # If it's a Conv2d output (B, C, H, W), we can get grid directly
        if embeddings.ndim == 4:
            _, _, h_grid, w_grid = embeddings.shape
            # Flatten: (B, C, H, W) -> (B, N, C)
            embeddings = embeddings.flatten(2).transpose(1, 2)
            num_patches = h_grid * w_grid
        else:
            # Fallback for already flattened inputs (less reliable for grid inference)
            _, num_patches, _ = embeddings.shape
            h_grid = int(math.sqrt(num_patches))
            w_grid = h_grid

        # 3. Check if we need interpolation
        if self.position_embedding.shape[0] == num_patches:
            return embeddings + self.position_embedding.unsqueeze(0)

        # 4. Interpolate Position Embeddings
        orig_num = int(self.position_embedding.shape[0])
        orig_grid = int(math.sqrt(orig_num))

        if orig_grid * orig_grid != orig_num:
            raise ValueError("Original positional embedding is not a square grid")

        # Reshape -> (1, Hidden, OrigH, OrigW)
        pos = self.position_embedding.view(orig_grid, orig_grid, -1).permute(2, 0, 1).unsqueeze(0)

        # Interpolate
        pos_resized = F.interpolate(pos, size=(h_grid, w_grid), mode="bilinear", align_corners=False)

        # Flatten back -> (N_new, Hidden)
        pos_resized = pos_resized.squeeze(0).permute(1, 2, 0).view(h_grid * w_grid, -1)

        # 5. Add and Return
        return embeddings + pos_resized.unsqueeze(0)
