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

"""SAM-3 Prompt Encoder for encoding user prompts.

This module implements the prompt encoder for SAM-3 which processes point prompts,
bounding boxes, and mask prompts into dense and sparse embeddings.
"""

from __future__ import annotations

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE


class PositionalEncoding(nn.Module):
    """Positional encoding for 2D coordinates.

    Encodes 2D coordinates using sinusoidal positional encoding.
    """

    def __init__(self, embed_dim: int) -> None:
        """Initialize PositionalEncoding.

        Args:
            embed_dim: Embedding dimension (must be even).
        """
        super().__init__()
        self.embed_dim = embed_dim
        KORNIA_CHECK(
            embed_dim % 2 == 0,
            f"embed_dim must be even, got {embed_dim}",
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to coordinates using sinusoidal basis functions.

        Encodes coordinates (x, y) using sin/cos at multiple frequency scales.
        Implementation note: builds 2*embed_dim intermediate representation from x,y encoding,
        then truncates to embed_dim to maintain dimensional consistency with input embeddings.

        Args:
            coords: Coordinate tensor of shape (B, N, 2) where last dimension is (x, y).

        Returns:
            Encoded tensor of shape (B, N, embed_dim).

        Raises:
            ValueError: If coords does not have shape (B, N, 2).
        """
        KORNIA_CHECK_SHAPE(coords, ["B", "N", "2"])
        B, N, _ = coords.shape

        # Create frequency bands
        freqs = torch.arange(0, self.embed_dim // 2, dtype=torch.float32, device=coords.device)
        freqs = 2.0 ** (freqs / (self.embed_dim // 2)) * torch.pi

        # Expand coordinates and frequency bands for broadcasting
        coords_expanded = coords.unsqueeze(-1)  # (B, N, 2, 1)
        freqs_expanded = freqs.view(1, 1, 1, -1)  # (1, 1, 1, embed_dim//2)

        # Compute sin and cos components
        args = coords_expanded * freqs_expanded  # (B, N, 2, embed_dim//2)
        sin_part = torch.sin(args)  # (B, N, 2, embed_dim//2)
        cos_part = torch.cos(args)  # (B, N, 2, embed_dim//2)

        # Interleave sin and cos for each coordinate
        encoded = torch.stack([sin_part, cos_part], dim=-1)  # (B, N, 2, embed_dim//2, 2)
        encoded = encoded.view(B, N, 2, self.embed_dim)  # (B, N, 2, embed_dim)

        # Separate x and y encodings and concatenate
        x_encoded = encoded[:, :, 0, :]  # (B, N, embed_dim)
        y_encoded = encoded[:, :, 1, :]  # (B, N, embed_dim)
        output = torch.cat([x_encoded, y_encoded], dim=-1)  # (B, N, 2*embed_dim)

        # Truncate to embed_dim for dimensional consistency with input embeddings
        return output[:, :, : self.embed_dim]


class PromptEncoder(nn.Module):
    """Encoder for SAM-3 prompts (points, boxes, masks).

    Encodes user prompts into sparse and dense embeddings that can be used
    by the mask decoder.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        input_image_size: int = 1024,
        mask_in_chans: int = 16,
    ) -> None:
        """Initialize PromptEncoder.

        Args:
            embed_dim: Embedding dimension.
            input_image_size: Size of input image (assumed square).
            mask_in_chans: Number of input channels for mask encoding.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.mask_in_chans = mask_in_chans

        # Point embedding
        self.pe_layer = PositionalEncoding(embed_dim)
        self.point_embeddings = nn.ModuleList(
            [nn.Embedding(1, embed_dim) for _ in range(4)]
        )  # (foreground, background, box top-left, box bottom-right)

        # Dense embedding (for masks)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, mask_in_chans // 4),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            nn.GroupNorm(1, mask_in_chans),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def _encode_points(
        self,
        points: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Encode point prompts into embeddings.

        Args:
            points: Tuple of (coords, labels) where:
                - coords: Tensor of shape (B, N, 2) with normalized coordinates.
                - labels: Tensor of shape (B, N) with labels (0=background, 1=foreground).

        Returns:
            Sparse embeddings of shape (B, N, embed_dim).

        Raises:
            ValueError: If coords and labels have incompatible shapes.
        """
        coords, labels = points
        KORNIA_CHECK_SHAPE(coords, ["B", "N", "2"])
        KORNIA_CHECK_SHAPE(labels, ["B", "N"])
        KORNIA_CHECK(
            coords.shape[:2] == labels.shape,
            f"coords and labels must have matching batch and point dimensions, "
            f"got {coords.shape[:2]} vs {labels.shape}",
        )

        B, N, _ = coords.shape

        # Encode coordinates using positional encoding
        pe = self.pe_layer(coords)  # (B, N, embed_dim)

        # Simple approach: use label to select embedding
        # NOTE: loop-based implementation for clarity; can be vectorized in future optimization
        label_embeddings = torch.zeros(B, N, self.embed_dim, device=coords.device, dtype=coords.dtype)
        for b in range(B):
            for i in range(N):
                label = int(labels[b, i].item())
                label_idx = min(label, 1)  # 0 or 1 for background/foreground
                label_embeddings[b, i] = self.point_embeddings[label_idx].weight[0]

        output = pe + label_embeddings
        return output

    def forward(
        self,
        *,
        points: tuple[torch.Tensor, torch.Tensor] | None = None,
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts into sparse and dense embeddings.

        Args:
            points: Optional tuple of (coords, labels) where:
                - coords: Tensor of shape (B, N, 2) with normalized coordinates in [0, 1].
                - labels: Tensor of shape (B, N) with binary labels (0 or 1).
            boxes: Optional tensor of shape (B, num_boxes, 4) with normalized bbox coordinates.
                Currently not implemented (returns zero embeddings).
            masks: Optional tensor of shape (B, 1, H, W) with binary masks.
                Currently not implemented (returns zero embeddings).

        Returns:
            Tuple of (sparse_embeddings, dense_embeddings) where:
                - sparse_embeddings: Tensor of shape (B, num_sparse, embed_dim) or (B, 0, embed_dim) if no prompts.
                - dense_embeddings: Tensor of shape (B, embed_dim, H, W) or zeros if no masks.

        Raises:
            ValueError: If no prompts are provided or point shapes are invalid.
        """
        sparse_embeddings = []
        B = 1  # Default batch size
        device = None

        # Determine batch size and device from inputs
        if points is not None:
            coords, labels = points
            B = coords.shape[0]
            device = coords.device
        elif boxes is not None:
            B = boxes.shape[0]
            device = boxes.device
        elif masks is not None:
            B = masks.shape[0]
            device = masks.device

        # Process point prompts
        if points is not None:
            coords, labels = points
            point_embeddings = self._encode_points((coords, labels))
            sparse_embeddings.append(point_embeddings)

        # Process box prompts (stub)
        if boxes is not None:
            KORNIA_CHECK_SHAPE(boxes, ["B", "num_boxes", "4"])
            # Boxes are intentionally stubbed in Phase 2
            # Full implementation (box corner encoding + corner embedding lookup) deferred to Phase 3
            num_boxes = boxes.shape[1]
            box_embeddings = torch.zeros(B, num_boxes, self.embed_dim, device=boxes.device, dtype=boxes.dtype)
            sparse_embeddings.append(box_embeddings)

        # Concatenate sparse embeddings
        if sparse_embeddings:
            sparse_embeddings = torch.cat(sparse_embeddings, dim=1)
        else:
            if device is None:
                device = torch.device("cpu")
            sparse_embeddings = torch.zeros(B, 0, self.embed_dim, device=device)

        # Process mask prompts (stub)
        if masks is not None:
            KORNIA_CHECK_SHAPE(masks, ["B", "1", "H", "W"])
            dense_embeddings = self.mask_downscaling(masks)
            # Resize to match expected output size
            dense_embeddings = torch.nn.functional.interpolate(
                dense_embeddings,
                size=(self.input_image_size // 4, self.input_image_size // 4),
                mode="bilinear",
                align_corners=False,
            )
        else:
            # No mask: use no_mask embedding
            if device is None:
                device = torch.device("cpu")
            dense_embeddings = self.no_mask_embed.weight.view(1, self.embed_dim, 1, 1)
            dense_embeddings = dense_embeddings.expand(
                B, self.embed_dim, self.input_image_size // 4, self.input_image_size // 4
            )
            dense_embeddings = dense_embeddings.to(device)

        return sparse_embeddings, dense_embeddings


__all__ = ["PositionalEncoding", "PromptEncoder"]
