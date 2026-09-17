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

"""Specialized sparse SIFT: one Gaussian pyramid for detection and description.

The pyramid, sparse DoG refinement, and descriptor share octave/layer conventions
and stay together here. The detection mechanics follow Lowe (2004) and OpenCV
SIFT. This variant deliberately omits contrast and edge rejection, selecting
valid extrema by top-K absolute response instead.

Descriptor support sizes and histogram weights follow OpenCV SIFT:
https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/sift.simd.hpp
Fixed 19/41 sample grids with bilinear gradient sampling replace integer-pixel
integration; gradients are transformed into LAF coordinates before voting.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from kornia.constants import pi
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.utils import _l2_normalize
from kornia.feature.laf import _grid_sample_patches, laf_from_center_scale_ori, laf_is_valid, rotate_laf
from kornia.feature.scale_space_detector import _check_mask, _resize_mask
from kornia.feature.siftdesc import _gradient_magnitude_orientation, _rootsift
from kornia.filters import get_gaussian_kernel1d, spatial_gradient
from kornia.geometry.subpix.spatial_soft_argmax import _solve_cramer_sym3x3


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
        # The detector promotes reduced precision inputs before pyramid construction.
        # Keep the convolution path for direct half-precision callers: its accumulation
        # order avoids a larger error across repeated pyramid levels.
        # Small arrays are faster in one convolution than in many slice additions.
        if (
            image.device.type == "cpu"
            and image.dtype not in (torch.float16, torch.bfloat16)
            and image.numel() >= 256 * 256
        ):
            return self._blur_cpu(image, kernel, radius)
        horizontal = F.conv2d(self._reflect_pad(image, radius, True), kernel.view(1, 1, 1, -1))
        return F.conv2d(self._reflect_pad(horizontal, radius, False), kernel.view(1, 1, -1, 1))

    def _blur_cpu(self, image: torch.Tensor, kernel: torch.Tensor, radius: int) -> torch.Tensor:
        """Apply a separable kernel without CPU convolution's im2col buffer."""

        def blur_axis(value: torch.Tensor, horizontal: bool) -> torch.Tensor:
            padded = self._reflect_pad(value, radius, horizontal)
            size = value.shape[-1 if horizontal else -2]
            if horizontal:
                result = padded[..., radius : radius + size] * kernel[radius]
                for offset in range(1, radius + 1):
                    left = padded[..., radius - offset : radius + size - offset]
                    right = padded[..., radius + offset : radius + size + offset]
                    if torch.compiler.is_compiling():
                        result = result + left * kernel[radius - offset] + right * kernel[radius + offset]
                    else:
                        result.add_(left, alpha=kernel[radius - offset])
                        result.add_(right, alpha=kernel[radius + offset])
            else:
                result = padded[..., radius : radius + size, :] * kernel[radius]
                for offset in range(1, radius + 1):
                    left = padded[..., radius - offset : radius + size - offset, :]
                    right = padded[..., radius + offset : radius + size + offset, :]
                    if torch.compiler.is_compiling():
                        result = result + left * kernel[radius - offset] + right * kernel[radius + offset]
                    else:
                        result.add_(left, alpha=kernel[radius - offset])
                        result.add_(right, alpha=kernel[radius + offset])
            return result

        return blur_axis(blur_axis(image, True), False)

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


class _SIFTScaleSpaceDetector(nn.Module):
    """Classical sparse DoG SIFT detector for a six-level Gaussian pyramid.

    The supplied ``scale_pyr`` returns Gaussian octaves shaped ``(B, 1, 6, H, W)``.
    This intentionally does not share the generic response/subpixel detector path:
    extrema and iterative subpixel refinement follow the SIFT algorithm. All
    valid strict extrema are ranked by absolute refined DoG value. There is
    deliberately no contrast threshold or Hessian edge rejection.
    """

    def __init__(self, num_features: int, scale_pyr: nn.Module) -> None:
        super().__init__()
        if num_features < 0:
            raise ValueError("num_features must be nonnegative")
        self.num_features = num_features
        self.scale_pyr = scale_pyr

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        # Like nn.Module.compile(), do not serialize a process-local compiled
        # callable. Unpickling restores the eager class method automatically.
        state.pop("_refine", None)
        return state

    @staticmethod
    def _values(dog: torch.Tensor, b: torch.Tensor, s: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return dog[b, s, y, x]

    @staticmethod
    def _neighbourhood(
        dog: torch.Tensor, b: torch.Tensor, s: torch.Tensor, y: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        # Share one gather between all finite differences / scale comparisons.
        # Index the flattened volume: gathering from an unfolded view would
        # allocate a dense (B, D-2, H-2, W-2, 3, 3, 3) tensor in backward.
        depth, height, width = dog.shape[1:]
        center = ((b * depth + s) * height + y) * width + x
        offsets = torch.tensor(
            [ds * height * width + dy * width + dx for ds in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
            device=dog.device,
            dtype=torch.long,
        )
        return dog.reshape(-1)[center[:, None] + offsets].reshape(-1, 3, 3, 3)

    def _refine(
        self, dog: torch.Tensor, b: torch.Tensor, s: torch.Tensor, y: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Iterative, sparse 3-D quadratic fit around strict DoG extrema."""
        d, h, w = dog.shape[1:]
        alive = torch.ones_like(s, dtype=torch.bool)
        shift_x = torch.zeros_like(s, dtype=dog.dtype)
        shift_y = torch.zeros_like(s, dtype=dog.dtype)
        shift_s = torch.zeros_like(s, dtype=dog.dtype)
        for _ in range(5):
            inside = (s > 0) & (s < d - 1) & (y >= 5) & (y < h - 5) & (x >= 5) & (x < w - 5)
            alive &= inside
            # Keep rejected rows addressable while active rows are solved; this
            # prevents a recentered dead candidate from indexing outside DoG.
            s = s.clamp(1, d - 2)
            y = y.clamp(1, h - 2)
            x = x.clamp(1, w - 2)
            if not alive.any():
                break
            values = self._neighbourhood(dog, b, s, y, x)
            c = values[:, 1, 1, 1]
            xp, xm = values[:, 1, 1, 2], values[:, 1, 1, 0]
            yp, ym = values[:, 1, 2, 1], values[:, 1, 0, 1]
            sp, sm = values[:, 2, 1, 1], values[:, 0, 1, 1]
            gx, gy, gs = (xp - xm) * 0.5, (yp - ym) * 0.5, (sp - sm) * 0.5
            dxx, dyy, dss = xp + xm - 2 * c, yp + ym - 2 * c, sp + sm - 2 * c
            dxy = (values[:, 1, 2, 2] - values[:, 1, 2, 0] - values[:, 1, 0, 2] + values[:, 1, 0, 0]) * 0.25
            dxs = (values[:, 2, 1, 2] - values[:, 2, 1, 0] - values[:, 0, 1, 2] + values[:, 0, 1, 0]) * 0.25
            dys = (values[:, 2, 2, 1] - values[:, 2, 0, 1] - values[:, 0, 2, 1] + values[:, 0, 0, 1]) * 0.25
            normalizer = (
                torch.stack([dxx, dyy, dss, dxy, dxs, dys, gx, gy, gs])
                .abs()
                .amax(0)
                .clamp_min(torch.finfo(dog.dtype).tiny)
            )
            sx, sy, ss, solved = _solve_cramer_sym3x3(
                dxx / normalizer,
                dyy / normalizer,
                dss / normalizer,
                dxy / normalizer,
                dxs / normalizer,
                dys / normalizer,
                -gx / normalizer,
                -gy / normalizer,
                -gs / normalizer,
            )
            alive = alive & solved & torch.isfinite(sx) & torch.isfinite(sy) & torch.isfinite(ss)
            shift_x, shift_y, shift_s = sx, sy, ss
            move = alive & ((sx.abs() >= 0.5) | (sy.abs() >= 0.5) | (ss.abs() >= 0.5))
            x = x + torch.where(move, sx.round().long(), torch.zeros_like(x))
            y = y + torch.where(move, sy.round().long(), torch.zeros_like(y))
            s = s + torch.where(move, ss.round().long(), torch.zeros_like(s))
            if not move.any():
                break
        converged = alive & (shift_x.abs() < 0.5) & (shift_y.abs() < 0.5) & (shift_s.abs() < 0.5)
        inside = (s >= 1) & (s < d - 1) & (x >= 5) & (x < w - 5) & (y >= 5) & (y < h - 5)
        converged = converged & inside
        shift_x = torch.where(converged, shift_x, torch.zeros_like(shift_x))
        shift_y = torch.where(converged, shift_y, torch.zeros_like(shift_y))
        shift_s = torch.where(converged, shift_s, torch.zeros_like(shift_s))
        return b, s.clamp(1, d - 2), y.clamp(1, h - 2), x.clamp(1, w - 2), shift_x, shift_y, shift_s, converged

    def _octave(self, gaussian: torch.Tensor, octave: int, mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        dog = (gaussian[:, 0, 1:] - gaussian[:, 0, :-1]).contiguous()
        _bsz, depth, height, width = dog.shape
        if depth < 3 or height < 11 or width < 11:
            empty = torch.empty(0, device=dog.device, dtype=torch.long)
            return empty, empty, empty, empty, dog.new_empty(0), dog.new_empty(0), dog.new_empty(0), dog.new_empty(0)
        # Four axial comparisons cheaply reject most pixels before gathering
        # complete 3-D neighbourhoods. The sparse pass below still requires a
        # strict extremum against all 26 neighbours, including spatial diagonals.
        searchable = dog[:, 1:-1]
        center = searchable[:, :, 5:-5, 5:-5]
        left, right = searchable[:, :, 5:-5, 4:-6], searchable[:, :, 5:-5, 6:-4]
        above, below = searchable[:, :, 4:-6, 5:-5], searchable[:, :, 6:-4, 5:-5]
        spatial_max = (center > left) & (center > right) & (center > above) & (center > below)
        spatial_min = (center < left) & (center < right) & (center < above) & (center < below)
        candidate_mask = spatial_max | spatial_min
        keep_map = None
        if mask is not None:
            keep_map = _resize_mask(mask, gaussian[:, :, 0]).squeeze(1)
            if keep_map.shape[0] == 1:
                keep_map = keep_map.expand(gaussian.shape[0], -1, -1)
            candidate_mask = candidate_mask & (keep_map[:, None, 5:-5, 5:-5] > 0)
        candidates = candidate_mask.nonzero()
        if candidates.numel() == 0:
            empty = torch.empty(0, device=dog.device, dtype=torch.long)
            return empty, empty, empty, empty, dog.new_empty(0), dog.new_empty(0), dog.new_empty(0), dog.new_empty(0)
        b, s, y, x = candidates.unbind(1)
        s, y, x = s + 1, y + 5, x + 5
        center = self._values(dog, b, s, y, x)
        neighbours = self._neighbourhood(dog, b, s, y, x).flatten(1)
        strict_max = (center[:, None] > neighbours[:, :13]).all(dim=1) & (center[:, None] > neighbours[:, 14:]).all(
            dim=1
        )
        strict_min = (center[:, None] < neighbours[:, :13]).all(dim=1) & (center[:, None] < neighbours[:, 14:]).all(
            dim=1
        )
        keep = strict_max | strict_min
        b, s, y, x = b[keep], s[keep], y[keep], x[keep]
        if b.numel() == 0:
            empty = torch.empty(0, device=dog.device, dtype=torch.long)
            return empty, empty, empty, empty, dog.new_empty(0), dog.new_empty(0), dog.new_empty(0), dog.new_empty(0)
        b, s, y, x, sx, sy, ss, good = self._refine(dog, b, s, y, x)
        c = self._values(dog, b, s, y, x)
        gx = (self._values(dog, b, s, y, x + 1) - self._values(dog, b, s, y, x - 1)) * 0.5
        gy = (self._values(dog, b, s, y + 1, x) - self._values(dog, b, s, y - 1, x)) * 0.5
        gs = (self._values(dog, b, s + 1, y, x) - self._values(dog, b, s - 1, y, x)) * 0.5
        contrast = c + 0.5 * (gx * sx + gy * sy + gs * ss)
        good = good & torch.isfinite(contrast)
        mask_weight = torch.ones_like(contrast)
        if keep_map is not None:
            x0, x1 = (x + sx).floor().long().clamp(0, width - 1), (x + sx).ceil().long().clamp(0, width - 1)
            y0, y1 = (y + sy).floor().long().clamp(0, height - 1), (y + sy).ceil().long().clamp(0, height - 1)
            mask_weight = torch.stack(
                [keep_map[b, y0, x0], keep_map[b, y0, x1], keep_map[b, y1, x0], keep_map[b, y1, x1]]
            ).amin(0)
            good &= mask_weight > 0
        pixel = 0.5 * float(2**octave)
        sigma = 1.6 * torch.exp2((s.to(dog.dtype) + ss) / 3.0) * pixel
        response = contrast.abs() * mask_weight
        b, s, y, x, response, sigma, sx, sy, sign = (
            item[good] for item in (b, s, y, x, response, sigma, sx, sy, contrast >= 0)
        )
        key = (((b * depth + s) * height + y) * width + x) * 2 + sign.long()
        order = key.argsort()
        keep = torch.ones_like(order, dtype=torch.bool)
        keep[1:] = key[order][1:] != key[order][:-1]
        order = order[keep]
        return b[order], s[order], y[order], x[order], response[order], sigma[order], sx[order], sy[order]

    def _detect_with_pyramid(
        self, img: torch.Tensor, num_feats: int, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        KORNIA_CHECK_SHAPE(img, ["B", "1", "H", "W"])
        if mask is not None:
            _check_mask(mask, img)
        bsz, dtype, device = img.shape[0], img.dtype, img.device
        if bsz == 0 or num_feats == 0:
            responses = img.new_zeros(bsz, num_feats)
            lafs = img.new_zeros(bsz, num_feats, 2, 3)
            filled = torch.zeros(bsz, num_feats, device=device, dtype=torch.bool)
            ids = torch.full((bsz, num_feats), -1, device=device, dtype=torch.long)
            return responses, lafs, filled, [img.unsqueeze(2)], ids, ids.clone()
        # The sparse fit uses second derivatives; keep them out of half-precision
        # underflow while preserving the public image/LAF/descriptor dtype.
        work = img.float() if dtype in (torch.float16, torch.bfloat16) else img
        pyramid = self.scale_pyr(work)
        entries = []
        for octave, gaussian in enumerate(pyramid):
            result = self._octave(gaussian, octave, mask)
            if result[0].numel() == 0:
                continue
            b, layer, y, x, response, sigma, sx, sy = result
            pixel = 0.5 * float(2**octave)
            xy = torch.stack([(x.to(gaussian.dtype) + sx) * pixel, (y.to(gaussian.dtype) + sy) * pixel], -1)
            entries.append((b, layer, xy, response, sigma, torch.full_like(layer, octave)))
        responses = torch.zeros(bsz, num_feats, device=device, dtype=dtype)
        lafs = torch.zeros(bsz, num_feats, 2, 3, device=device, dtype=dtype)
        filled = torch.zeros(bsz, num_feats, device=device, dtype=torch.bool)
        octaves = torch.full((bsz, num_feats), -1, device=device, dtype=torch.long)
        levels = torch.full_like(octaves, -1)
        if num_feats == 0 or not entries:
            return responses, lafs, filled, pyramid, octaves, levels
        b = torch.cat([entry[0] for entry in entries])
        layer = torch.cat([entry[1] for entry in entries])
        xy = torch.cat([entry[2] for entry in entries])
        response = torch.cat([entry[3] for entry in entries])
        sigma = torch.cat([entry[4] for entry in entries])
        octave = torch.cat([entry[5] for entry in entries])
        for batch in range(bsz):
            idx = (b == batch).nonzero().flatten()
            if idx.numel() == 0:
                continue
            take = idx[response[idx].topk(min(num_feats, idx.numel())).indices]
            count = take.numel()
            scale = (6.0 * sigma[take]).view(1, count, 1, 1)
            lafs[batch : batch + 1, :count] = laf_from_center_scale_ori(xy[take].view(1, count, 2), scale)[0]
            responses[batch, :count] = response[take]
            filled[batch, :count] = True
            octaves[batch, :count], levels[batch, :count] = octave[take], layer[take]
        return responses, lafs, filled, pyramid, octaves, levels

    def forward(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-padded LAFs and absolute DoG responses."""
        responses, lafs, _, _, _, _ = self._detect_with_pyramid(img, self.num_features, mask)
        return lafs, responses


class _SIFTScaleSpaceDescriptor(nn.Module):
    """Orient and describe LAFs from their detector-selected Gaussian levels.

    ``pyramid[octave]`` is ``(B, 1, L, H, W)`` and ``octave_indices`` and
    ``level_indices`` identify each LAF's converged integer Gaussian level. Coordinates in
    ``lafs`` are in original-image pixels; the supplied octave index sets the
    pixel distance to ``0.5 * 2**octave`` for the standard doubled SIFT pyramid.
    """

    def __init__(self, rootsift: bool = True, upright: bool = False, clipval: float = 0.2) -> None:
        super().__init__()
        self.rootsift, self.upright, self.clipval = rootsift, upright, clipval
        self.eps = 1e-10

    @staticmethod
    def _grid(
        lafs: torch.Tensor, layer_indices: torch.Tensor, level_height: int, width: int, atlas_height: int, size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = (2.0 * torch.arange(size, dtype=lafs.dtype, device=lafs.device) + 1.0) / size - 1.0
        yy, xx = torch.meshgrid(q, q, indexing="ij")
        local = torch.stack([xx, yy], -1)
        points = torch.einsum("bnij,xyj->bnxyi", lafs[..., :2, :2], local) + lafs[..., :2, 2].view(
            lafs.shape[0], lafs.shape[1], 1, 1, 2
        )
        # Clamp in the selected layer before shifting it into the vertical atlas:
        # atlas-border padding would otherwise interpolate a neighbouring layer.
        x = points[..., 0].clamp(0.0, float(width - 1))
        y = points[..., 1].clamp(0.0, float(level_height - 1))
        y = y + layer_indices.view(lafs.shape[0], lafs.shape[1], 1, 1).to(y.dtype) * level_height
        grid = torch.stack([2.0 * (x + 0.5) / width - 1.0, 2.0 * (y + 0.5) / atlas_height - 1.0], -1)
        return grid.reshape(lafs.shape[0], lafs.shape[1] * size, size, 2), xx, yy

    def _sample_gradients(
        self, gradients: torch.Tensor, lafs: torch.Tensor, layer_indices: torch.Tensor, level_height: int, size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        atlas_height, w = gradients.shape[-2:]
        work_dtype = torch.float32 if gradients.dtype in (torch.float16, torch.bfloat16) else gradients.dtype
        work_lafs = lafs.to(work_dtype)
        grid, xx, yy = self._grid(work_lafs, layer_indices, level_height, w, atlas_height, size)
        sampled = _grid_sample_patches(gradients.to(work_dtype), grid, atlas_height, w)
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

    def _orientation(
        self, gradients: torch.Tensor, lafs: torch.Tensor, layer_indices: torch.Tensor, level_height: int
    ) -> torch.Tensor:
        mag, angle, xx, yy = self._sample_gradients(gradients, lafs, layer_indices, level_height, 19)
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

    def _describe(
        self, gradients: torch.Tensor, lafs: torch.Tensor, layer_indices: torch.Tensor, level_height: int
    ) -> torch.Tensor:
        mag, angle, xx, yy = self._sample_gradients(gradients, lafs, layer_indices, level_height, 41)
        desc = self._descriptor_histograms(mag, angle, xx, yy)
        desc = _l2_normalize(desc, dim=-1).clamp(0.0, self.clipval)
        desc = _l2_normalize(desc, dim=-1)
        return _rootsift(desc.reshape(-1, 128), self.eps).reshape_as(desc) if self.rootsift else desc

    @staticmethod
    def _descriptor_histograms(
        mag: torch.Tensor, angle: torch.Tensor, xx: torch.Tensor, yy: torch.Tensor
    ) -> torch.Tensor:
        """Accumulate the two angular votes per sample into the 4 x 4 spatial cells."""
        b, n, _ = mag.shape
        weight = torch.exp(-0.78125 * (xx.square() + yy.square())).to(mag.dtype)
        angular = (angle % (2 * pi)) * 8 / (2 * pi)
        lower = angular.floor()
        fraction = angular - lower
        lower = lower.long() % 8
        weighted = mag * weight
        # Only two angular bins have nonzero weight. Avoid broadcasting every
        # sample against all eight bins and materializing their distances.
        # MPS matmul benefits from contiguous (bins, samples) matrices; CPU
        # scatter is faster with each sample's eight bins next to one another.
        if mag.device.type == "mps":
            angular_weights = mag.new_zeros(b, n, 8, mag.shape[-1]).transpose(-1, -2)
        else:
            angular_weights = mag.new_zeros(b, n, mag.shape[-1], 8)
        angular_weights.scatter_(-1, lower.unsqueeze(-1), (weighted * (1.0 - fraction)).unsqueeze(-1))
        angular_weights.scatter_add_(-1, ((lower + 1) % 8).unsqueeze(-1), (weighted * fraction).unsqueeze(-1))
        spatial_x = 2.5 * xx + 1.5
        spatial_y = 2.5 * yy + 1.5
        bins = torch.arange(4, device=mag.device, dtype=mag.dtype)
        cell_y, cell_x = torch.meshgrid(bins, bins, indexing="ij")
        spatial_weights = (1.0 - (spatial_x.unsqueeze(-1) - cell_x.reshape(-1)).abs()).clamp_min(0.0) * (
            1.0 - (spatial_y.unsqueeze(-1) - cell_y.reshape(-1)).abs()
        ).clamp_min(0.0)
        # (B, N, 8, samples) @ (samples, 16) -> one 8-bin histogram per spatial cell.
        return torch.matmul(angular_weights.transpose(-1, -2), spatial_weights).transpose(-1, -2).reshape(b, n, 128)

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
            octave_selected = valid & (octave_indices == octave)
            if not octave_selected.any():
                continue
            used_levels = torch.unique(level_indices[octave_selected], sorted=True)
            used_images = octave_images.index_select(2, used_levels).permute(0, 2, 1, 3, 4)
            level_height, width = used_images.shape[-2:]
            gradients = spatial_gradient(used_images.reshape(-1, 1, level_height, width).to(work_dtype), "diff")
            gradients = gradients[:, 0].reshape(b, used_levels.numel(), 2, level_height, width)
            atlas = gradients.permute(0, 2, 1, 3, 4).reshape(b, 2, used_levels.numel() * level_height, width)
            # Keep CPU voting temporaries in cache; accelerators benefit from
            # larger chunks that amortize kernel launches and synchronization.
            chunk_size = 128 if image.device.type == "cpu" else 1024
            for batch in range(b):
                indices = octave_selected[batch].nonzero().flatten()
                for start in range(0, indices.numel(), chunk_size):
                    current_indices = indices[start : start + chunk_size]
                    if current_indices.numel() == 0:
                        continue
                    layer_indices = torch.searchsorted(used_levels, level_indices[batch, current_indices]).view(1, -1)
                    current = safe[batch : batch + 1, current_indices].clone()
                    current[..., :2, :] /= pixel
                    if not (self.upright if upright is None else upright):
                        orientation_lafs = current.clone()
                        orientation_lafs[..., :2, :2] *= 0.75
                        alpha = self._orientation(
                            atlas[batch : batch + 1], orientation_lafs, layer_indices, level_height
                        )
                        current = rotate_laf(current, torch.rad2deg(alpha).unsqueeze(-1))
                    descriptor_lafs = current.clone()
                    descriptor_lafs[..., :2, :2] *= 1.25
                    oriented[batch, current_indices] = current[0] * pixel
                    descriptors[batch, current_indices] = self._describe(
                        atlas[batch : batch + 1], descriptor_lafs, layer_indices, level_height
                    )[0]
        return torch.where(valid.view(b, n, 1, 1), oriented, lafs.to(oriented)).to(lafs.dtype), descriptors.masked_fill(
            ~valid.unsqueeze(-1), 0.0
        ).to(output_dtype)
