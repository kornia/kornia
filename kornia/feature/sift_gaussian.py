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

"""Fixed Gaussian scale space for the specialized SIFT pipeline."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from kornia.filters import get_gaussian_kernel1d


class _SIFTScalePyramid(nn.Module):
    """Six Gaussian images per octave with exact factor-two coordinate spacing.

    The input camera sigma is 0.5 pixels. Doubling makes it 1.0; initial
    smoothing reaches sigma 1.6. Three intervals per octave require five
    incremental blurs, with Gaussian level three seeding the next octave.
    """

    def __init__(self) -> None:
        super().__init__()
        step = 2.0 ** (1.0 / 3.0)
        sigmas = [math.sqrt(1.6**2 - 1.0)]
        sigmas += [1.6 * step**i * math.sqrt(step**2 - 1.0) for i in range(5)]
        for index, sigma in enumerate(sigmas):
            size = int(8.0 * sigma + 1.0) | 1
            kernel = get_gaussian_kernel1d(size, sigma, dtype=torch.float64).float().reshape(-1)
            self.register_buffer(f"kernel_{index}", kernel)

    @staticmethod
    def _double(image: torch.Tensor) -> torch.Tensor:
        # Size 2H-1 maps even output pixels exactly onto original pixels. Pad
        # the final half-pixel with border replication, as precise 2x sampling.
        height, width = image.shape[-2:]
        doubled = F.interpolate(image, size=(2 * height - 1, 2 * width - 1), mode="bilinear", align_corners=True)
        return F.pad(doubled, (0, 1, 0, 1), mode="replicate")

    @staticmethod
    def _reflect_pad(image: torch.Tensor, radius: int, horizontal: bool) -> torch.Tensor:
        size = image.shape[-1 if horizontal else -2]
        if radius < size:
            padding = (radius, radius, 0, 0) if horizontal else (0, 0, radius, radius)
            return F.pad(image, padding, mode="reflect")
        # Reflection remains defined on tiny octaves even when PyTorch's pad
        # primitive rejects a radius >= the side length.
        index = torch.arange(-radius, size + radius, device=image.device)
        if size == 1:
            index = torch.zeros_like(index)
        else:
            period = 2 * (size - 1)
            index = index.remainder(period)
            index = torch.minimum(index, period - index)
        return image.index_select(-1 if horizontal else -2, index)

    def _blur(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        kernel = kernel.to(image)
        radius = kernel.numel() // 2
        horizontal = F.conv2d(self._reflect_pad(image, radius, True), kernel.view(1, 1, 1, -1))
        return F.conv2d(self._reflect_pad(horizontal, radius, False), kernel.view(1, 1, -1, 1))

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Build doubled-image Gaussian octaves for normalized grayscale images."""
        first = self._blur(self._double(image), self.kernel_0)
        pyramid = []
        while True:
            levels = [first]
            for index in range(1, 6):
                levels.append(self._blur(levels[-1], getattr(self, f"kernel_{index}")))
            pyramid.append(torch.stack(levels, dim=2))
            height, width = first.shape[-2:]
            if min(height // 2, width // 2) < 12:
                break
            # Decimate the already anti-aliased sigma=3.2 level. Bilinear
            # resizing here would shift pixel centres and add untracked blur.
            first = levels[3][..., : 2 * (height // 2) : 2, : 2 * (width // 2) : 2]
        return pyramid
