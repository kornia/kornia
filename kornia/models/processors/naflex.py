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
from torch import Tensor, nn
from torch.nn import functional as F

from kornia.core._compat import torch_version_lt

__all__ = ["NaFlex"]


class NaFlex(nn.Module):
    r"""NaFlex wrapper for vision embeddings that supports flexible image sizes.

    This module wraps a patch embedding function and a square position embedding table. It computes the patch
    embeddings of an image and adds the table resized to the image's patch grid, as transformers'
    ``Siglip2VisionEmbeddings.resize_positional_embeddings`` does for SigLIP 2's NaFlex variant: the resize is
    bilinear interpolation (``align_corners=False``), antialiased when the grid is smaller than the table along an
    axis, so a downsampled axis averages the table instead of sampling it. transformers always antialiases; when no
    axis is downsampled antialiasing changes nothing (up to float rounding), and the plain resize keeps working where
    the antialiased one does not (see the note). The table is added unresized only when the patch grid is the table's
    own square grid.

    Args:
        patch_embedding_fcn: A callable (e.g., Conv2d) that takes images and returns
            patch embeddings of shape :math:`(B, C, h, w)` or :math:`(B, N, C)`. A :math:`(B, N, C)` output is read
            as the grid of :math:`N` cells with the aspect ratio of the :math:`H \times W` image,
            :math:`h = \mathrm{round}(\sqrt{N H / W})` rows of :math:`w = N / h` patches in raster order.
        position_embedding: Position embedding table of shape :math:`(S^2, C)`, the raster-order rows of an
            :math:`S \times S` grid, to be interpolated to match input resolution. An ``nn.Parameter`` stays a
            parameter of the module, so it is trained with it; any other tensor is registered as a buffer. A table
            that is not a square grid is added as is when it has one row per patch.

    Note:
        A float16 or bfloat16 table on the CPU is resized in float32 and cast back, as transformers does: the CPU has
        no half-precision antialiased bilinear kernel. A table on MPS that is downsampled is resized on the CPU in
        float32 with torch older than 2.7, which has no MPS antialiased kernel. The dynamo ONNX exporter
        (``torch.onnx.export(..., dynamo=True)``) exports the antialiased resize; the legacy TorchScript exporter
        cannot (``UnsupportedOperatorError`` for ``aten::_upsample_bilinear2d_aa``), so it exports only grids at least
        as large as the table along both axes.

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
        if isinstance(position_embedding, nn.Parameter):
            self.position_embedding = position_embedding
        else:
            self.register_buffer("position_embedding", position_embedding)

    def forward(self, pixel_values: Tensor) -> Tensor:
        r"""Forward pass through NaFlex wrapper.

        Args:
            pixel_values: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            Embeddings with interpolated positional embeddings of shape :math:`(B, N, C)`.

        Raises:
            ValueError: If original positional embedding does not form a square grid and does not have one row per
                patch, or if a :math:`(B, N, C)` patch output does not form a grid of the image's aspect ratio.
        """
        embeddings = self.patch_embedding_fcn(pixel_values)
        if embeddings.ndim == 4:
            _, _, h_grid, w_grid = embeddings.shape
            embeddings = embeddings.flatten(2).transpose(1, 2)
        else:
            num_patches = embeddings.shape[1]
            height, width = pixel_values.shape[-2:]
            h_grid = max(1, round(math.sqrt(num_patches * height / width)))
            w_grid = num_patches // h_grid
            if h_grid * w_grid != num_patches:
                raise ValueError(
                    f"Cannot read the {num_patches} patch embeddings of a {height} x {width} image as a grid: the grid "
                    f"of that aspect ratio, {h_grid} x {w_grid}, has {h_grid * w_grid} cells."
                )
        position_embedding = self.position_embedding
        orig_num = int(position_embedding.shape[0])
        orig_grid = math.isqrt(orig_num)
        if orig_grid * orig_grid != orig_num:
            if orig_num == h_grid * w_grid:
                return embeddings + position_embedding.unsqueeze(0)
            raise ValueError(f"Original positional embedding is not a square grid (got {orig_num} embeddings)")
        if (h_grid, w_grid) == (orig_grid, orig_grid):
            return embeddings + position_embedding.unsqueeze(0)
        # transformers always passes antialias=True. Antialiasing only changes a downsampled axis: an upsampled one
        # gets plain bilinear weights either way (equal up to float rounding), so the plain resize, which the legacy
        # TorchScript ONNX exporter and MPS before torch 2.7 support, is used when no axis shrinks. Python ints from
        # the shapes: dynamo guards on the comparison instead of breaking the graph.
        antialias = h_grid < orig_grid or w_grid < orig_grid
        pos = position_embedding.reshape(orig_grid, orig_grid, -1).permute(2, 0, 1).unsqueeze(0)
        # Required, not an optimization: the CPU has no float16/bfloat16 antialiased bilinear kernel
        # (NotImplementedError on torch 2.14, RuntimeError on 2.5.1), as transformers notes where it upcasts on the CPU
        # (for every resize, which keeps its float32 values), and MPS has none before torch 2.7. A float64 table keeps
        # its precision.
        if (pos.device.type == "cpu" and pos.dtype in (torch.float16, torch.bfloat16)) or (
            antialias and pos.device.type == "mps" and torch_version_lt(2, 7, 0)
        ):
            pos = pos.to("cpu", torch.float32)
        pos_resized = F.interpolate(
            pos, size=(h_grid, w_grid), mode="bilinear", align_corners=False, antialias=antialias
        )
        pos_resized = pos_resized.squeeze(0).permute(1, 2, 0).reshape(h_grid * w_grid, -1)
        pos_resized = pos_resized.to(position_embedding.device, position_embedding.dtype)

        return embeddings + pos_resized.unsqueeze(0)
