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
from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kornia.core._compat import torch_version_lt
from kornia.core.check import KORNIA_CHECK
from kornia.core.utils import _warn_at_caller, is_compiling

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
            patch embeddings of shape :math:`(B, C, h, w)` or :math:`(B, N, C)`. A :math:`(B, C, h, w)` output gives
            the patch grid. A :math:`(B, N, C)` output carries no grid: without ``patch_size`` its :math:`N` tokens
            are read as a square :math:`\sqrt{N} \times \sqrt{N}` grid in raster order, and a :math:`N` that is not
            a square number raises. A non-square grid therefore needs ``patch_size`` or a :math:`(B, C, h, w)` output:
            the :math:`8 \times 32` grid of a :math:`128 \times 512` image at patch 16 is read right with
            ``patch_size=16``, and as :math:`16 \times 16` without it. A square reading that takes rectangular patches,
            on an image whose sides differ by at least the patch height or width of its longer side, warns with a
            ``UserWarning`` naming ``patch_size`` (outside ``torch.compile`` and ``torch.export``), since it is wrong
            for square patches; ``patch_size`` silences it, also for a patch embedding that is rectangular. A
            square-patch convolution padded by at most half its kernel never warns.
        position_embedding: Position embedding table of shape :math:`(S^2, C)`, the raster-order rows of an
            :math:`S \times S` grid, to be interpolated to match input resolution. An ``nn.Parameter`` stays a
            parameter of the module, so it is trained with it; any other tensor is registered as a buffer. A table
            that is not a square grid is added as is when it has one row per patch.
        patch_size: The patch size :math:`p` or :math:`(p_h, p_w)` of a patch embedding that returns
            :math:`(B, N, C)`, which then has the grid :math:`(H // p_h, W // p_w)` of whole patches with the remainder
            dropped (a ``Conv2d`` with that kernel and stride); the grid must have :math:`N` cells. Ignored for a
            :math:`(B, C, h, w)` output. Default: ``None``, the square reading above.

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
        patch_size: Optional[Union[int, tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        self.patch_embedding_fcn = patch_embedding_fcn
        self.patch_size: Optional[tuple[int, int]] = None
        if patch_size is not None:
            self.patch_size = (
                (patch_size, patch_size) if isinstance(patch_size, int) else (patch_size[0], patch_size[1])
            )
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
                patch.
            BaseError: If a :math:`(B, N, C)` patch output's grid cannot be read: with ``patch_size``, when
                :math:`(H // p_h) (W // p_w) \neq N`; without it, when :math:`N` is not a square number.
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
        h_grid, w_grid = grid if grid is not None else self._token_grid(num_patches, *pixel_values.shape[-2:])
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

    def _token_grid(self, num_patches: int, height: int, width: int) -> tuple[int, int]:
        """Return the patch grid of the ``num_patches`` tokens of a ``height x width`` image.

        ``(height // p_h, width // p_w)`` with ``patch_size``, else the square grid of a square ``num_patches``.

        Raises:
            BaseError: if that grid does not have ``num_patches`` cells, or without ``patch_size`` if
                ``num_patches`` is not a square number.
        """
        # The checks format the sizes only on failure: an f-string evaluated on every call makes Dynamo specialize the
        # image size, so a dynamic-shape torch.compile would recompile for each new size.
        if self.patch_size is not None:
            patch_h, patch_w = self.patch_size
            h_grid, w_grid = height // patch_h, width // patch_w
            KORNIA_CHECK(
                h_grid * w_grid == num_patches,
                lambda: (
                    f"NaFlex reads the grid of a (B, N, C) patch output from patch_size={self.patch_size}: a "
                    f"{height} x {width} image has a {h_grid} x {w_grid} grid of {h_grid * w_grid} patches, but the "
                    f"patch embedding returned {num_patches}."
                ),
            )
            return h_grid, w_grid
        side = math.isqrt(num_patches)
        KORNIA_CHECK(
            side * side == num_patches,
            lambda: (
                f"NaFlex reads a (B, N, C) patch output without patch_size as a square grid, but {num_patches} "
                "patch embeddings are not a square number. Pass patch_size to NaFlex, or make the patch embedding "
                "return (B, C, h, w)."
            ),
        )
        # A square grid on an image whose sides differ by at least one patch of the longer side takes rectangular
        # patches. That is right for a rectangular patch embedding, but for square patches the grid is not square
        # (8 x 32 at patch 16 for 128 x 512) and the table goes in the wrong layout. N, H and W cannot tell the
        # two apart, so warn. A square-patch convolution of stride s, overlapping or not, with padding up to half its
        # kernel never warns: its grid is square only for sides H < W less than s apart, and H >= (side - 1) * s, so
        # (W - H) * side < (side - 1) * s + (W - H) <= W.
        if not is_compiling() and abs(height - width) * side >= max(height, width):
            _warn_at_caller(
                f"NaFlex reads the {num_patches} patch embeddings of a {height} x {width} image as a {side} x {side} "
                f"grid, which takes {height / side:.3g} x {width / side:.3g} pixel patches. If the patches are square, "
                "the grid is not square and the position table is added in the wrong layout: pass patch_size to "
                "NaFlex (which also silences this warning), or make the patch embedding return (B, C, h, w)."
            )
        return side, side
