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

import torch

from kornia.geometry.subpix import nms2d

from .structs import Keypoints


def heatmap_to_keypoints(
    heatmap: torch.Tensor, n: int | None = None, window_size: int = 5, score_threshold: float = 0.0
) -> list[Keypoints]:
    """Detect keypoints as the strict local maxima of a heatmap, in raster order.

    Args:
        heatmap: detection logits of shape ``(B, 1, H, W)``.
        n: the maximum number of keypoints per image, counted among the maxima above ``score_threshold``. An image
            with ``n`` such maxima or fewer keeps all of them; with more, its ``n`` strongest, where maxima tied at the
            ``n``-th score are taken in raster order (row by row, then column by column) until ``n`` are kept.
            ``None`` keeps every maximum.
        window_size: the size of the non-maxima suppression window.
        score_threshold: a maximum is kept only if its score is strictly above it.

    Returns:
        A list of length :math:`B` with the keypoints of each image.

    """
    heatmap = heatmap.squeeze(1)
    nms_mask = nms2d(heatmap.unsqueeze(1), (window_size, window_size), mask_only=True).squeeze(1)
    nmsed = nms_mask & (heatmap > score_threshold)

    width = heatmap.shape[-1]
    keypoints = []
    for b in range(heatmap.shape[0]):
        # The raster indices of the maxima are the image's one read of the device; everything after is gathered by
        # index, where boolean masks would read it again for each tensor.
        flat = torch.nonzero(nmsed[b].flatten()).squeeze(-1)
        detection_logp = heatmap[b].flatten()[flat]

        if n is not None and flat.numel() > n:
            # The n strongest: every score above the n-th largest (torch.kthvalue gives it as the n-th smallest of
            # -logp), then the maxima equal to it in raster order until n are kept. The keypoints stay in raster order.
            if n > 0:
                minus_threshold, _indices = torch.kthvalue(-detection_logp, n)
                threshold = -minus_threshold
                above = detection_logp > threshold
                tied = detection_logp == threshold
                mask = above | (tied & (torch.cumsum(tied, 0) <= n - above.sum()))
                # Exactly n are kept, so their positions come from the running count of the mask without reading
                # the device: the i-th kept maximum goes to slot i, every other one to a spare slot n.
                slot = torch.where(mask, torch.cumsum(mask, 0) - 1, n)
                order = torch.zeros(n + 1, dtype=torch.long, device=flat.device)
                order.scatter_(0, slot, torch.arange(flat.numel(), device=flat.device))
                kept = order[:n]
            else:
                kept = flat[:0]
            flat = flat[kept]
            detection_logp = detection_logp[kept]

        xy = torch.stack((flat % width, torch.div(flat, width, rounding_mode="floor")), dim=-1)
        keypoints.append(Keypoints(xy, detection_logp))

    return keypoints
