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
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["NaFlex"]


class NaFlex(nn.Module):
    """NaFlex wrapper for vision embeddings that supports flexible image sizes.

    This wraps a vision embedding module (e.g., :class:`SigLip2VisionEmbeddings`) and
    adapts its position embeddings to input images whose grid size differs from the
    original configuration by resizing the positional grid with bilinear interpolation.

    Args:
        base: The base module that provides a convolutional patch embedding and a
            position embedding attribute.
        embedding_attr: Name of the position embedding attribute on ``base``.
    """

    def __init__(self, base: Any, embedding_attr: str = "position_embedding") -> None:
        super().__init__()
        self.base = base
        self.embedding_attr = embedding_attr

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Ensure base provides the requested embedding attribute at call time
        if not hasattr(self.base, self.embedding_attr):
            raise AttributeError(f"Base object has no attribute '{self.embedding_attr}'")

        pos_embed = getattr(self.base, self.embedding_attr)

        # Prefer using a conv patch extractor if available, otherwise try calling base
        if hasattr(self.base, "patch_embedding") and isinstance(self.base.patch_embedding, nn.Conv2d):
            # Extract patches via conv, then flatten like SigLip2VisionEmbeddings
            embeddings = self.base.patch_embedding(pixel_values)  # (B, hidden, H', W')
            _, _, h_grid, w_grid = embeddings.shape
            embeddings = embeddings.flatten(2).transpose(1, 2)  # (B, N, hidden)
            num_patches = h_grid * w_grid
        else:
            # Fallback: try calling base directly and assume it returns (B, N, hidden)
            out = self.base(pixel_values)
            # If base returns a tuple (as some models do), pick the embeddings
            if isinstance(out, tuple):
                embeddings = out[1] if len(out) > 1 else out[0]
            else:
                embeddings = out
            _, num_patches, _ = embeddings.shape
            # infer grid
            h_grid = int(math.sqrt(num_patches))
            w_grid = h_grid

        # If pos_embed length matches, add directly
        if pos_embed.shape[0] == num_patches:
            embeddings = embeddings + pos_embed.unsqueeze(0)
            return embeddings

        # Otherwise, resize positional embedding from original grid to current grid
        orig_num = int(pos_embed.shape[0])
        orig_grid = int(math.sqrt(orig_num))
        if orig_grid * orig_grid != orig_num:
            raise ValueError("Original positional embedding is not square grid")

        # reshape pos_embed -> (1, hidden, orig_grid, orig_grid)
        pos = pos_embed.view(orig_grid, orig_grid, -1).permute(2, 0, 1).unsqueeze(0)
        # interpolate to (h_grid, w_grid)
        pos_resized = F.interpolate(pos, size=(h_grid, w_grid), mode="bilinear", align_corners=False)
        # back to (orig_hidden -> hidden), (h_grid * w_grid, hidden)
        pos_resized = pos_resized.squeeze(0).permute(1, 2, 0).view(h_grid * w_grid, -1)

        # If hidden dims mismatch, try to broadcast or raise
        if pos_resized.shape[1] != embeddings.shape[2]:
            raise ValueError("Positional embedding hidden size does not match embedding hidden size")

        embeddings = embeddings + pos_resized.unsqueeze(0)
        return embeddings
