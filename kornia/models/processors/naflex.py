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

from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["NaFlex"]


class NaFlex(nn.Module):
    r"""NaFlex wrapper for vision embeddings that supports flexible image sizes.

    This module wraps a patch embedding function and a position embedding tensor.
    It computes patch embeddings and dynamically interpolates the position embeddings
    to match the input resolution using bilinear interpolation.

    Args:
        patch_embedding_fcn: A callable (e.g., Conv2d) that takes images and returns
            patch embeddings of shape :math:`(B, C, H, W)` or :math:`(B, N, C)`.
        position_embedding: Position embedding tensor of shape :math:`(N, C)` to be
            interpolated to match input resolution.

    Example:
        >>> import torch
        >>> from kornia.models.processors.naflex import NaFlex
        >>> patch_fcn = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16)
        >>> pos_embed = torch.randn(196, 768)  # 14x14 grid
        >>> model = NaFlex(patch_fcn, pos_embed)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([1, 196, 768])
    """

    def __init__(
        self,
        patch_embedding_fcn: Callable[[Tensor], Tensor],
        position_embedding: Tensor,
    ) -> None:
        super().__init__()
        self.patch_embedding_fcn = patch_embedding_fcn
        self.register_buffer("position_embedding", position_embedding)

    def forward(self, pixel_values: Tensor) -> Tensor:
        r"""Forward pass through NaFlex wrapper.

        Args:
            pixel_values: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            Embeddings with interpolated positional embeddings of shape :math:`(B, N, C)`.

        Raises:
            ValueError: If original positional embedding does not form a square grid.
        """
        embeddings = self.patch_embedding_fcn(pixel_values)
        if embeddings.ndim == 4:
            _, _, h_grid, w_grid = embeddings.shape
            embeddings = embeddings.flatten(2).transpose(1, 2)
            num_patches = h_grid * w_grid
        else:
            _, num_patches, _ = embeddings.shape
            h_grid = int(math.sqrt(num_patches))
            w_grid = h_grid
        if self.position_embedding.shape[0] == num_patches:
            return embeddings + self.position_embedding.unsqueeze(0)
        orig_num = int(self.position_embedding.shape[0])
        orig_grid = int(math.sqrt(orig_num))
        if orig_grid * orig_grid != orig_num:
            raise ValueError(f"Original positional embedding is not a square grid (got {orig_num} embeddings)")
        pos = self.position_embedding.view(orig_grid, orig_grid, -1).permute(2, 0, 1).unsqueeze(0)
        pos_resized = F.interpolate(pos, size=(h_grid, w_grid), mode="bilinear", align_corners=False)
        pos_resized = pos_resized.squeeze(0).permute(1, 2, 0).view(h_grid * w_grid, -1)

        return embeddings + pos_resized.unsqueeze(0)
