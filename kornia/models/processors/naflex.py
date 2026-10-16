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
from kornia.core.check import KORNIA_CHECK

__all__ = ["NaFlex"]


class NaFlex(nn.Module):
    r"""NaFlex wrapper for vision embeddings that supports flexible image sizes.

    This module wraps a patch embedding function and a square position embedding table. It computes the patch
    embeddings of an image and adds the table resized to the image's patch grid, as transformers'
    ``Siglip2VisionEmbeddings.resize_positional_embeddings`` does for SigLIP 2's NaFlex variant: the resize is
    bilinear interpolation (``align_corners=False``), antialiased when the grid is smaller than the table along an
    axis, so a downsampled axis averages the table instead of sampling it. transformers always antialiases; when no
    axis is downsampled antialiasing changes nothing (up to float rounding), and the plain resize keeps working where
    the antialiased one does not (see the note). A square table is added unresized only when the patch grid is the
    table's own grid.

    Args:
        patch_embedding_fcn: A callable (e.g., Conv2d) that takes images and returns
            patch embeddings of shape :math:`(B, C, h, w)` or :math:`(B, N, C)`. A :math:`(B, N, C)` output carries
            no grid, so it is read from the :math:`H \times W` image, in raster order. First as the grid of
            :math:`N` cells with the image's aspect ratio, :math:`h = \mathrm{round}(\sqrt{N H / W})` rows of
            :math:`w = N / h` patches, used when it has :math:`N` cells and some integer :math:`p` gives
            :math:`H // p = h` and :math:`W // p = w`: that is the grid of whole :math:`p \times p` patches with the
            remainder dropped (a ``Conv2d`` with kernel and stride :math:`p`), always so when H and W are multiples of
            the patch size. Otherwise, when :math:`N` is a square number, as the square grid
            :math:`\sqrt{N} \times \sqrt{N}`, which covers square-grid embeddings such as a strided convolution or
            non-square patches tiling a non-square image; any other :math:`N` raises. :math:`N` alone cannot tell two
            embeddings with the same token count apart: non-square patches whose grid is not the square-patch
            reading, e.g. :math:`8 \times 32` patches of a :math:`128 \times 512` image (a :math:`16 \times 16`
            grid), are read as the square-patch grid :math:`8 \times 32`. Such an embedding should return
            :math:`(B, C, h, w)`, whose grid is not inferred. With square patches and sides that are not patch
            multiples, the square fallback can also differ from the true grid (e.g. :math:`2 \times 2` patches of an
            :math:`8 \times 3` image, a :math:`4 \times 1` grid, are read as :math:`2 \times 2`).
        position_embedding: Position embedding table of shape :math:`(S^2, C)`, the raster-order rows of an
            :math:`S \times S` grid, to be interpolated to match input resolution. An ``nn.Parameter`` stays a
            parameter of the module, so it is trained with it; any other tensor is registered as a buffer. A table
            that is not a square grid is added as is when it has one row per patch.

    Note:
        A float16 or bfloat16 table on the CPU is resized in float32 and cast back, as transformers does: the CPU has
        no half-precision antialiased bilinear kernel. A table on MPS that is downsampled is resized on the CPU in
        float32 with torch older than 2.7, which has no MPS antialiased kernel.

        Eager mode picks the resize per call, from the grid and the table size. An exported graph with dynamic H and W
        keeps the choice made for its example input. With the dynamo ONNX exporter
        (``torch.onnx.export(..., dynamo=True)``), export from an input whose grid is smaller than the table along both
        axes: the graph then resizes with antialiasing at every size, which is right for every grid because
        antialiasing changes nothing when upsampling. ``torch.export.export`` reports that size comparison as a
        constraint violation; ``torch.onnx.export`` exports anyway on torch 2.14 and refuses on torch 2.5.1. The
        legacy TorchScript exporter cannot export the antialiased resize (``UnsupportedOperatorError`` for
        ``aten::_upsample_bilinear2d_aa``), so it exports only from an example grid at least as large as the table
        along both axes, and its graph resizes without antialiasing at every size, which is exact only for such grids.

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
                patch, or if the :math:`N` tokens of a :math:`(B, N, C)` patch output form no grid of the image's
                aspect ratio and :math:`N` is not a square number.
            BaseError: If the grid of the image's aspect ratio has :math:`N` cells but no integer patch size
                :math:`p` gives it (:math:`H // p = h` and :math:`W // p = w`), and :math:`N` is not a square number.
        """
        embeddings = self.patch_embedding_fcn(pixel_values)
        grid = embeddings.shape[2:] if embeddings.ndim == 4 else None
        if embeddings.ndim == 4:
            embeddings = embeddings.flatten(2).transpose(1, 2)
        num_patches = embeddings.shape[1]
        position_embedding = self.position_embedding
        orig_num = int(position_embedding.shape[0])
        orig_grid = math.isqrt(orig_num)
        if orig_grid * orig_grid != orig_num:
            # a table that is not a square grid cannot be resized; one row per patch is added in raster order
            if orig_num == num_patches:
                return embeddings + position_embedding.unsqueeze(0)
            raise ValueError(f"Original positional embedding is not a square grid (got {orig_num} embeddings)")
        h_grid, w_grid = grid if grid is not None else self._read_grid(num_patches, *pixel_values.shape[-2:])
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

    @staticmethod
    def _read_grid(num_patches: int, height: int, width: int) -> tuple[int, int]:
        """Read the patch grid of ``num_patches`` tokens of a ``height x width`` image.

        The grid of the image's aspect ratio when it has ``num_patches`` cells and some integer patch size ``p`` gives
        it, else the square grid when ``num_patches`` is a square number.

        Raises:
            ValueError: if no grid of the aspect ratio has ``num_patches`` cells and ``num_patches`` is not square.
            BaseError: if the grid of the aspect ratio has ``num_patches`` cells, no integer patch size ``p`` gives it
                (``height // p = h`` and ``width // p = w``), and ``num_patches`` is not square.
        """
        h_grid = max(1, round(math.sqrt(num_patches * height / width)))
        w_grid = num_patches // h_grid
        factorizes = h_grid * w_grid == num_patches
        consistent = False
        if factorizes:
            # The aspect-ratio reading can transpose the grid of an image whose sides are not multiples of the patch
            # size (48 x 47 at patch 16 has the grid 3 x 2 and is read as 2 x 3). It is the grid (H // p, W // p) of a
            # p x p patching exactly when some integer p gives H // p == h and W // p == w: p must lie in
            # (H / (h + 1), H / h] and in (W / (w + 1), W / w].
            p_min = max(height // (h_grid + 1), width // (w_grid + 1)) + 1
            p_max = min(height // h_grid, width // w_grid)
            consistent = p_min <= p_max
            if consistent:
                return h_grid, w_grid
        # Otherwise a square number of tokens is read as the square grid, as NaFlex always read a (B, N, C) output: it
        # is right for square-grid embeddings that no whole p x p patching explains, such as a stride-2 3 x 3
        # convolution (9 x 9 tokens of a 20 x 20 image) or (16, 32) patches of a 256 x 512 image (16 x 16 tokens).
        side = math.isqrt(num_patches)
        if side * side == num_patches:
            return side, side
        if not factorizes:
            raise ValueError(
                f"Cannot read the {num_patches} patch embeddings of a {height} x {width} image as a grid: the grid "
                f"of that aspect ratio, {h_grid} x {w_grid}, has {h_grid * w_grid} cells, and {num_patches} is not a "
                "square number."
            )
        KORNIA_CHECK(
            consistent,
            f"NaFlex reads the grid of a (B, N, C) patch output from the image shape: {num_patches} patch embeddings "
            f"of a {height} x {width} image read as a {h_grid} x {w_grid} grid, which no whole patch size p gives "
            f"({height} // p == {h_grid} and {width} // p == {w_grid}), and {num_patches} is not a square number. Use "
            "an image whose sides are multiples of the patch size, or a patch embedding that returns (B, C, h, w).",
        )
        return h_grid, w_grid
