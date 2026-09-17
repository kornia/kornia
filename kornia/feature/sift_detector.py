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
# limitations under the License.

"""Sparse classical SIFT DoG detector (Lowe 2004; OpenCV SIFT)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.subpix import nms2d
from kornia.geometry.subpix.spatial_soft_argmax import _solve_cramer_sym3x3

from .laf import laf_from_center_scale_ori
from .scale_space_detector import _check_mask, _resize_mask


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

    @staticmethod
    def _values(dog: torch.Tensor, b: torch.Tensor, s: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return dog[b, s, y, x]

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
            c = self._values(dog, b, s, y, x)
            xp, xm = self._values(dog, b, s, y, x + 1), self._values(dog, b, s, y, x - 1)
            yp, ym = self._values(dog, b, s, y + 1, x), self._values(dog, b, s, y - 1, x)
            sp, sm = self._values(dog, b, s + 1, y, x), self._values(dog, b, s - 1, y, x)
            gx, gy, gs = (xp - xm) * 0.5, (yp - ym) * 0.5, (sp - sm) * 0.5
            dxx, dyy, dss = xp + xm - 2 * c, yp + ym - 2 * c, sp + sm - 2 * c
            dxy = (
                self._values(dog, b, s, y + 1, x + 1)
                - self._values(dog, b, s, y + 1, x - 1)
                - self._values(dog, b, s, y - 1, x + 1)
                + self._values(dog, b, s, y - 1, x - 1)
            ) * 0.25
            dxs = (
                self._values(dog, b, s + 1, y, x + 1)
                - self._values(dog, b, s + 1, y, x - 1)
                - self._values(dog, b, s - 1, y, x + 1)
                + self._values(dog, b, s - 1, y, x - 1)
            ) * 0.25
            dys = (
                self._values(dog, b, s + 1, y + 1, x)
                - self._values(dog, b, s + 1, y - 1, x)
                - self._values(dog, b, s - 1, y + 1, x)
                + self._values(dog, b, s - 1, y - 1, x)
            ) * 0.25
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
        # Restrict the dense pass to the three searchable Gaussian intervals.
        # Strict 2-D NMS avoids max-pool index buffers and drops plateaus before
        # sparse cross-scale comparisons or quadratic fitting allocate anything.
        searchable = dog[:, 1:-1].reshape(-1, 1, height, width)
        spatial_max = nms2d(searchable, (3, 3), mask_only=True).reshape(_bsz, depth - 2, height, width)
        spatial_min = nms2d(-searchable, (3, 3), mask_only=True).reshape_as(spatial_max)
        candidate_mask = spatial_max | spatial_min
        keep_map = None
        if mask is not None:
            keep_map = _resize_mask(mask, gaussian[:, :, 0]).squeeze(1)
            if keep_map.shape[0] == 1:
                keep_map = keep_map.expand(gaussian.shape[0], -1, -1)
            candidate_mask = candidate_mask & (keep_map[:, None] > 0)
        candidates = candidate_mask[:, :, 5:-5, 5:-5].nonzero()
        if candidates.numel() == 0:
            empty = torch.empty(0, device=dog.device, dtype=torch.long)
            return empty, empty, empty, empty, dog.new_empty(0), dog.new_empty(0), dog.new_empty(0), dog.new_empty(0)
        b, s, y, x = candidates.unbind(1)
        s, y, x = s + 1, y + 5, x + 5
        center = self._values(dog, b, s, y, x)
        strict_max = spatial_max[b, s - 1, y, x]
        strict_min = spatial_min[b, s - 1, y, x]
        for ds in (-1, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbour = self._values(dog, b, s + ds, y + dy, x + dx)
                    strict_max &= center > neighbour
                    strict_min &= center < neighbour
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
