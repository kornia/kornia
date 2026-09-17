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

"""Quality-oriented sparse SIFT extraction from a detector Gaussian pyramid.

The support sizes and histogram weights follow OpenCV SIFT:
https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/sift.simd.hpp
Unlike its integer-pixel integration, this implementation uses fixed 19/41
sample grids with bilinear gradient sampling. Gradients are evaluated on the
selected Gaussian image, transformed into the LAF coordinates, then voted with
linear spatial and angular interpolation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kornia.constants import pi
from kornia.core.utils import _l2_normalize
from kornia.filters import spatial_gradient

from .laf import _grid_sample_patches, laf_is_valid, rotate_laf
from .siftdesc import _gradient_magnitude_orientation, _rootsift


class _SIFTScaleSpaceDescriptor(nn.Module):
    """Orient and describe LAFs from their detector-selected Gaussian levels.

    ``pyramid[octave]`` is ``(B, 1, L, H, W)`` and ``octave_indices`` and
    ``level_indices`` identify the level which produced each LAF. Coordinates in
    ``lafs`` are in original-image pixels; the supplied octave index sets the
    pixel distance to ``0.5 * 2**octave`` for the standard doubled SIFT pyramid.
    """

    def __init__(self, rootsift: bool = True, upright: bool = False, clipval: float = 0.2) -> None:
        super().__init__()
        self.rootsift, self.upright, self.clipval = rootsift, upright, clipval
        self.eps = 1e-10

    @staticmethod
    def _grid(
        lafs: torch.Tensor, height: int, width: int, size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = (2.0 * torch.arange(size, dtype=lafs.dtype, device=lafs.device) + 1.0) / size - 1.0
        yy, xx = torch.meshgrid(q, q, indexing="ij")
        local = torch.stack([xx, yy], -1)
        points = torch.einsum("bnij,xyj->bnxyi", lafs[..., :2, :2], local) + lafs[..., :2, 2].view(
            lafs.shape[0], lafs.shape[1], 1, 1, 2
        )
        grid = 2.0 * (points + 0.5) / points.new_tensor([width, height]) - 1.0
        return grid.reshape(lafs.shape[0], lafs.shape[1] * size, size, 2), xx, yy

    def _sample_gradients(
        self, gradients: torch.Tensor, lafs: torch.Tensor, size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = gradients.shape[-2:]
        work_dtype = torch.float32 if gradients.dtype in (torch.float16, torch.bfloat16) else gradients.dtype
        work_lafs = lafs.to(work_dtype)
        grid, xx, yy = self._grid(work_lafs, h, w, size)
        sampled = _grid_sample_patches(gradients.to(work_dtype), grid, h, w)
        sampled = sampled.reshape(lafs.shape[0], 2, lafs.shape[1], size * size).permute(0, 2, 1, 3)
        local_gradient = torch.einsum("bnji,bnjs->bnis", work_lafs[..., :2, :2], sampled)
        mag, angle = _gradient_magnitude_orientation(local_gradient[:, :, 0], local_gradient[:, :, 1], self.eps)
        return mag, angle, xx.reshape(-1), yy.reshape(-1)

    @staticmethod
    def _angular_histogram(mag: torch.Tensor, angle: torch.Tensor, bins: int) -> torch.Tensor:
        position = (angle % (2.0 * pi)) * bins / (2.0 * pi)
        lower = position.floor().long() % bins
        weight1 = position - position.floor()
        output = mag.new_zeros(*mag.shape[:-1], bins)
        output.scatter_add_(-1, lower, mag * (1.0 - weight1))
        output.scatter_add_(-1, (lower + 1) % bins, mag * weight1)
        return output

    def _orientation(self, gradients: torch.Tensor, lafs: torch.Tensor) -> torch.Tensor:
        mag, angle, xx, yy = self._sample_gradients(gradients, lafs, 19)
        # sigma = LAF scale / (6 * octave pixel size); expressed in local LAF
        # coordinates it is scale-independent, while retaining the stated scale law.
        radius2 = xx.square() + yy.square()
        weight = torch.exp(-4.5 * radius2).to(mag.dtype)
        histogram = self._angular_histogram(mag * weight, angle, 36)
        histogram = F.conv1d(
            F.pad(histogram.reshape(-1, 1, 36), (2, 2), mode="circular"),
            mag.new_tensor([1, 4, 6, 4, 1]).view(1, 1, 5) / 16,
        ).squeeze(1)
        index = histogram.argmax(-1)
        left = histogram.gather(1, ((index - 1) % 36).unsqueeze(1)).squeeze(1)
        center = histogram.gather(1, index.unsqueeze(1)).squeeze(1)
        right = histogram.gather(1, ((index + 1) % 36).unsqueeze(1)).squeeze(1)
        denominator = left + right - 2 * center
        safe = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        offset = torch.where(denominator != 0, 0.5 * (left - right) / safe, torch.zeros_like(denominator))
        return (-2.0 * pi * (index.to(lafs.dtype) + offset) / 36.0).reshape(lafs.shape[:2])

    def _describe(self, gradients: torch.Tensor, lafs: torch.Tensor) -> torch.Tensor:
        mag, angle, xx, yy = self._sample_gradients(gradients, lafs, 41)
        b, n, samples = mag.shape
        weight = torch.exp(-0.78125 * (xx.square() + yy.square())).to(mag.dtype)
        angular = (angle % (2 * pi)) * 8 / (2 * pi)
        a0 = angular.floor().long() % 8
        aw1 = angular - angular.floor()
        spatial_x = 2.5 * xx + 1.5
        spatial_y = 2.5 * yy + 1.5
        x0, y0 = spatial_x.floor().long(), spatial_y.floor().long()
        wx1, wy1 = spatial_x - x0, spatial_y - y0
        output = mag.new_zeros(b, n, 4, 4, 8)
        vote = mag * weight
        for dy, wy in ((0, 1.0 - wy1), (1, wy1)):
            for dx, wx in ((0, 1.0 - wx1), (1, wx1)):
                x_index, y_index = x0 + dx, y0 + dy
                valid = (x_index >= 0) & (x_index < 4) & (y_index >= 0) & (y_index < 4)
                spatial = (y_index.clamp(0, 3) * 4 + x_index.clamp(0, 3)) * 8
                for da, wa in ((0, 1.0 - aw1), (1, aw1)):
                    index = spatial.view(1, 1, samples) + (a0 + da) % 8
                    output.view(b, n, -1).scatter_add_(2, index, vote * wx * wy * wa * valid.view(1, 1, samples))
        desc = output.flatten(2)
        desc = _l2_normalize(desc, dim=-1).clamp(0.0, self.clipval)
        desc = _l2_normalize(desc, dim=-1)
        return _rootsift(desc.reshape(-1, 128), self.eps).reshape_as(desc) if self.rootsift else desc

    def forward(
        self,
        pyramid: list[torch.Tensor],
        lafs: torch.Tensor,
        octave_indices: torch.Tensor,
        level_indices: torch.Tensor,
        upright: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return oriented LAFs and 128D descriptors from selected pyramid levels."""
        b, n = lafs.shape[:2]
        image = pyramid[0]
        output_dtype = image.dtype
        work_dtype = torch.float32 if output_dtype in (torch.float16, torch.bfloat16) else output_dtype
        valid = laf_is_valid(lafs)
        safe = torch.where(
            valid.view(b, n, 1, 1), lafs.to(work_dtype), torch.eye(2, 3, device=image.device, dtype=work_dtype)
        )
        oriented, descriptors = safe.clone(), torch.zeros(b, n, 128, device=image.device, dtype=work_dtype)
        for octave, octave_images in enumerate(pyramid):
            pixel = 0.5 * float(2**octave)
            for level in range(octave_images.shape[2]):
                selected = valid & (octave_indices == octave) & (level_indices == level)
                if not selected.any():
                    continue
                level_image = octave_images[:, :, level].to(work_dtype)
                gradients = spatial_gradient(level_image, "diff")[:, 0]
                for batch in range(b):
                    indices = selected[batch].nonzero().flatten()
                    for start in range(0, indices.numel(), 256):
                        current_indices = indices[start : start + 256]
                        if current_indices.numel() == 0:
                            continue
                        current = safe[batch : batch + 1, current_indices].clone()
                        current[..., :2, :] /= pixel
                        if not (self.upright if upright is None else upright):
                            orientation_lafs = current.clone()
                            orientation_lafs[..., :2, :2] *= 0.75
                            alpha = self._orientation(gradients[batch : batch + 1], orientation_lafs)
                            current = rotate_laf(current, torch.rad2deg(alpha).unsqueeze(-1))
                        descriptor_lafs = current.clone()
                        descriptor_lafs[..., :2, :2] *= 1.25
                        oriented[batch, current_indices] = current[0] * pixel
                        descriptors[batch, current_indices] = self._describe(
                            gradients[batch : batch + 1], descriptor_lafs
                        )[0]
        return torch.where(valid.view(b, n, 1, 1), oriented, lafs.to(oriented)).to(lafs.dtype), descriptors.masked_fill(
            ~valid.unsqueeze(-1), 0.0
        ).to(output_dtype)
