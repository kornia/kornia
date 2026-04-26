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

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "Qwen3VLImageProcessor",
    "Qwen3VLImageProcessorConfig",
    "smart_resize",
]


_QWEN_VL_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
_QWEN_VL_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class Qwen3VLImageProcessorConfig:
    """Configuration for the Qwen3-VL dynamic-resolution image preprocessor.

    Args:
        patch_size: Spatial patch size used by the vision tower.
        spatial_merge_size: Post-encoder spatial merge factor; output spatial
            dimensions are constrained to multiples of
            ``patch_size * spatial_merge_size`` so each merged token corresponds
            to a whole patch grid cell.
        min_pixels: Minimum allowed number of input pixels (height * width)
            after resizing. Smaller inputs are upsampled.
        max_pixels: Maximum allowed number of input pixels after resizing.
            Larger inputs are downsampled.
        image_mean: Per-channel mean used for normalization.
        image_std: Per-channel standard deviation used for normalization.
        max_aspect_ratio: Inputs whose longer side exceeds this multiple of the
            shorter side are rejected with a :class:`ValueError`.
    """

    patch_size: int = 14
    spatial_merge_size: int = 2
    min_pixels: int = 56 * 56
    max_pixels: int = 28 * 28 * 1280
    image_mean: tuple[float, float, float] = field(default_factory=lambda: _QWEN_VL_MEAN)
    image_std: tuple[float, float, float] = field(default_factory=lambda: _QWEN_VL_STD)
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
    """Compute target spatial dimensions matching the Qwen3-VL resize policy.

    The policy returns dimensions divisible by ``factor`` whose product lies in
    ``[min_pixels, max_pixels]`` while keeping the original aspect ratio as
    closely as possible. This is the same procedure used by the Qwen2-VL and
    Qwen3-VL HuggingFace preprocessors, reproduced here in pure PyTorch.

    Args:
        height: Input height in pixels (must be positive).
        width: Input width in pixels (must be positive).
        factor: Required divisor for the output dimensions; typically
            ``patch_size * spatial_merge_size``.
        min_pixels: Lower bound for ``height * width`` of the result.
        max_pixels: Upper bound for ``height * width`` of the result.
        max_aspect_ratio: Reject inputs whose longer side exceeds this multiple
            of the shorter side.

    Returns:
        ``(target_height, target_width)`` — both ``>= factor`` and divisible
        by ``factor``.

    Raises:
        ValueError: If ``height`` or ``width`` is non-positive, ``factor`` is
            non-positive, ``min_pixels > max_pixels``, or the aspect ratio
            exceeds ``max_aspect_ratio``.
    """
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


class Qwen3VLImageProcessor(nn.Module):
    """Differentiable image preprocessor for Qwen3-VL.

    Performs three operations in order:

    1. Smart-resize the input to a shape divisible by
       ``patch_size * spatial_merge_size`` while keeping the pixel count in the
       configured ``[min_pixels, max_pixels]`` range.
    2. Bicubic interpolation (with anti-aliasing) implements the resize.
    3. Per-channel normalization with the configured ``image_mean`` /
       ``image_std`` buffers.

    The output is always ``BCHW`` so it can flow directly into
    :class:`Qwen3VLVisionTransformer`. All ops are differentiable, making the
    preprocessor usable inside training pipelines.

    Args:
        config: :class:`Qwen3VLImageProcessorConfig` controlling the resize
            policy and normalization statistics.
    """

    def __init__(self, config: Qwen3VLImageProcessorConfig) -> None:
        super().__init__()
        if len(config.image_mean) != len(config.image_std):
            raise ValueError(
                "image_mean and image_std must have the same length, got "
                f"{len(config.image_mean)} and {len(config.image_std)}."
            )
        self.config = config
        mean = torch.tensor(config.image_mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(config.image_std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    @property
    def factor(self) -> int:
        """Required divisor for output spatial dimensions."""
        return self.config.patch_size * self.config.spatial_merge_size

    def target_size(self, height: int, width: int) -> tuple[int, int]:
        """Return the resize target for an input of shape ``(height, width)``."""
        return smart_resize(
            height,
            width,
            self.factor,
            self.config.min_pixels,
            self.config.max_pixels,
            self.config.max_aspect_ratio,
        )

    def forward(self, images: Tensor) -> Tensor:
        """Resize and normalize a batch of images.

        Args:
            images: ``(B, C, H, W)`` tensor. All images in the batch share
                the same spatial shape; per-image dynamic resolution is
                supported by calling the processor once per shape group.

        Returns:
            ``(B, C, H', W')`` where ``(H', W') = target_size(H, W)``, with
            per-channel normalization applied.
        """
        if images.dim() != 4:
            raise ValueError(f"Expected 4D BCHW input; got shape {tuple(images.shape)}.")
        if images.shape[1] != self.image_mean.shape[1]:
            raise ValueError(
                f"Input has {images.shape[1]} channels but processor was configured for "
                f"{self.image_mean.shape[1]} (image_mean/std length)."
            )
        h, w = int(images.shape[-2]), int(images.shape[-1])
        target_h, target_w = self.target_size(h, w)
        if (target_h, target_w) != (h, w):
            images = F.interpolate(
                images,
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        mean = self.image_mean.to(images.dtype)
        std = self.image_std.to(images.dtype)
        return (images - mean) / std
