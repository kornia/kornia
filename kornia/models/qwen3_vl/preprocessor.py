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
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "Qwen3VLImageProcessor",
    "Qwen3VLImageProcessorConfig",
    "Qwen3VLPreprocessorOutput",
    "smart_resize",
]


_QWEN3_VL_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
_QWEN3_VL_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass
class Qwen3VLImageProcessorConfig:
    """Configuration for the Qwen3-VL dynamic-resolution image preprocessor.

    Defaults match the published ``Qwen/Qwen3-VL-2B-Instruct`` checkpoint:
    ``patch_size=16``, ``temporal_patch_size=2``, ``spatial_merge_size=2``,
    pixel range ``[256*256, 4096*4096]`` and centred-half normalisation.
    """

    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    in_channels: int = 3
    min_pixels: int = 256 * 256
    max_pixels: int = 4096 * 4096
    image_mean: tuple[float, float, float] = field(default_factory=lambda: _QWEN3_VL_MEAN)
    image_std: tuple[float, float, float] = field(default_factory=lambda: _QWEN3_VL_STD)
    rescale_factor: float = 1.0 / 255.0
    max_aspect_ratio: float = 200.0


def _round_by(value: float, factor: int) -> int:
    return int(round(value / factor) * factor)


def _floor_by(value: float, factor: int) -> int:
    return int(math.floor(value / factor) * factor)


def _ceil_by(value: float, factor: int) -> int:
    return int(math.ceil(value / factor) * factor)


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    max_aspect_ratio: float = 200.0,
) -> tuple[int, int]:
    """Compute target spatial dimensions matching the Qwen3-VL resize policy."""
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got ({height}, {width}).")
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}.")
    if min_pixels > max_pixels:
        raise ValueError(f"min_pixels ({min_pixels}) must not exceed max_pixels ({max_pixels}).")
    longer = max(height, width)
    shorter = min(height, width)
    if longer / shorter > max_aspect_ratio:
        raise ValueError(f"aspect ratio {longer / shorter:.2f} exceeds max_aspect_ratio={max_aspect_ratio}.")

    h_bar = max(_round_by(height, factor), factor)
    w_bar = max(_round_by(width, factor), factor)
    pixels = h_bar * w_bar

    if pixels > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(_floor_by(height / beta, factor), factor)
        w_bar = max(_floor_by(width / beta, factor), factor)
    elif pixels < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by(height * beta, factor)
        w_bar = _ceil_by(width * beta, factor)

    return h_bar, w_bar


class Qwen3VLPreprocessorOutput(NamedTuple):
    """Pair returned by :class:`Qwen3VLImageProcessor`.

    Holds ``(pixel_values, grid_thw)``: ``pixel_values`` are the flat patches
    expected by the vision tower and ``grid_thw`` is an integer tensor of
    shape ``(B, 3)`` describing each image's patch grid.
    """

    pixel_values: Tensor
    grid_thw: Tensor


class Qwen3VLImageProcessor(nn.Module):
    """Differentiable image preprocessor for Qwen3-VL.

    Performs (1) smart-resize to dimensions divisible by
    ``patch_size * spatial_merge_size``, (2) optional rescale + per-channel
    normalisation, and (3) patchification into the
    ``(num_patches, in_channels * temporal_patch_size * patch_size * patch_size)``
    layout consumed by :class:`Qwen3VLVisionModel`. All ops are differentiable.
    """

    def __init__(self, config: Qwen3VLImageProcessorConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = Qwen3VLImageProcessorConfig()
        if len(config.image_mean) != len(config.image_std):
            raise ValueError(
                "image_mean and image_std must have the same length, got "
                f"{len(config.image_mean)} and {len(config.image_std)}."
            )
        if config.in_channels != len(config.image_mean):
            raise ValueError(
                f"in_channels={config.in_channels} does not match len(image_mean)={len(config.image_mean)}."
            )
        self.config = config
        mean = torch.tensor(config.image_mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(config.image_std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    @property
    def factor(self) -> int:
        return self.config.patch_size * self.config.spatial_merge_size

    def target_size(self, height: int, width: int) -> tuple[int, int]:
        return smart_resize(
            height, width, self.factor, self.config.min_pixels, self.config.max_pixels, self.config.max_aspect_ratio
        )

    def forward(self, images: Tensor, do_rescale: bool = False) -> Qwen3VLPreprocessorOutput:
        """Resize, normalise, and patchify a batch of images.

        Args:
            images: ``(B, C, H, W)`` tensor. All images in the batch share the
                same spatial shape.
            do_rescale: If ``True``, multiply the input by ``rescale_factor``
                (default ``1/255``) before normalising. Set this when the
                input is in the ``[0, 255]`` range.

        Returns:
            :class:`Qwen3VLPreprocessorOutput` carrying ``pixel_values`` of
            shape ``(B * grid_h * grid_w, C * temporal_patch_size * patch_size**2)``
            and ``grid_thw`` of shape ``(B, 3)``.
        """
        if images.dim() != 4:
            raise ValueError(f"Expected 4D BCHW input; got shape {tuple(images.shape)}.")
        if images.shape[1] != self.config.in_channels:
            raise ValueError(
                f"Input has {images.shape[1]} channels but processor was configured for {self.config.in_channels}."
            )

        h, w = int(images.shape[-2]), int(images.shape[-1])
        target_h, target_w = self.target_size(h, w)
        if (target_h, target_w) != (h, w):
            images = F.interpolate(
                images, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True
            )

        if do_rescale:
            images = images * self.config.rescale_factor

        mean = self.image_mean.to(images.dtype)
        std = self.image_std.to(images.dtype)
        images = (images - mean) / std

        return self._patchify(images)

    def _patchify(self, images: Tensor) -> Qwen3VLPreprocessorOutput:
        cfg = self.config
        b, _c, h, w = images.shape
        grid_h = h // cfg.patch_size
        grid_w = w // cfg.patch_size
        if grid_h * cfg.patch_size != h or grid_w * cfg.patch_size != w:
            raise ValueError(f"Image size ({h}, {w}) is not divisible by patch_size={cfg.patch_size}.")

        # Add a temporal axis and repeat to fill `temporal_patch_size` frames so each
        # image becomes a single grid_t=1 video chunk.
        frames = images.unsqueeze(1).repeat(1, cfg.temporal_patch_size, 1, 1, 1)
        grid_t = 1

        # Layout follows transformers' Qwen2VLImageProcessorFast: outer block index
        # then intra-block index, with patch content laid out C, T, P_h, P_w.
        merge = cfg.spatial_merge_size
        if grid_h % merge != 0 or grid_w % merge != 0:
            raise ValueError(f"Patch grid ({grid_h}, {grid_w}) must be divisible by spatial_merge_size={merge}.")
        patches = frames.view(
            b,
            grid_t,
            cfg.temporal_patch_size,
            cfg.in_channels,
            grid_h // merge,
            merge,
            cfg.patch_size,
            grid_w // merge,
            merge,
            cfg.patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flat = patches.reshape(
            b * grid_t * grid_h * grid_w,
            cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size,
        )
        grid_thw = torch.tensor([[grid_t, grid_h, grid_w]] * b, dtype=torch.int64, device=images.device)
        return Qwen3VLPreprocessorOutput(flat, grid_thw)
