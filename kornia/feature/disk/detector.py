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

from typing import Optional

import torch
import torch.nn.functional as F

from kornia.core import Tensor

from .structs import Keypoints


def nms(signal: Tensor, window_size: int = 5, cutoff: float = 0.0) -> Tensor:
    """Apply non-maximum suppression."""
    if window_size % 2 != 1:
        raise ValueError(f"window_size has to be odd, got {window_size}")

    _, ixs = F.max_pool2d(signal, kernel_size=window_size, stride=1, padding=window_size // 2, return_indices=True)

    h, w = signal.shape[1:]
    coords = torch.arange(h * w, device=signal.device).reshape(1, h, w)
    nms = ixs == coords

    if cutoff is None:
        return nms
    else:
        return nms & (signal > cutoff)


def heatmap_to_keypoints(
    heatmap: Tensor, n: Optional[int] = None, window_size: int = 5, score_threshold: float = 0.0
) -> list[Keypoints]:
    """Inference-time nms-based detection protocol."""
    heatmap = heatmap.squeeze(1)  # (B, H, W)
    B, H, W = heatmap.shape
    device = heatmap.device

    nms_mask = nms(heatmap, window_size=window_size, cutoff=score_threshold)

    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    coords = torch.stack((x, y), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

    coords_flat = coords[nms_mask]
    scores_flat = heatmap[nms_mask]
    batch_indices = torch.arange(B, device=device).view(-1, 1, 1).expand(-1, H, W)[nms_mask]

    keypoints = []
    for b in range(B):
        mask = batch_indices == b
        if not mask.any():
            keypoints.append(Keypoints(coords_flat.new_zeros((0, 2)), scores_flat.new_zeros((0,))))
            continue

        xys = coords_flat[mask]
        detection_logp = scores_flat[mask]

        if n is not None and detection_logp.numel() > n:
            topk = torch.topk(detection_logp, k=n, largest=True, sorted=False)
            xys = xys[topk.indices]
            detection_logp = topk.values

        keypoints.append(Keypoints(xys, detection_logp))

    return keypoints
