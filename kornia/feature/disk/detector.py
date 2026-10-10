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
import torch.nn.functional as F

from .structs import Keypoints


def _nms_mask(heatmap: torch.Tensor, window_size: int) -> torch.Tensor:
    """Non-maxima suppression of DISK's reference implementation (cvlab-epfl/disk ``nms``).

    A position is kept when it is the (first) maximum of the ``window_size`` square window centred on it. The window
    is clipped at the image border, so a maximum on the first or last rows and columns is kept. Of several equal
    maxima in a window only the one the max-pool reports first survives.

    Args:
        heatmap: heatmap of shape :math:`(B, 1, H, W)`.
        window_size: odd side of the window in pixels.

    Returns:
        Boolean mask of shape :math:`(B, 1, H, W)`.
    """
    if window_size % 2 != 1:
        raise ValueError(f"window_size has to be odd, got {window_size}")
    _, ixs = F.max_pool2d(heatmap, kernel_size=window_size, stride=1, padding=window_size // 2, return_indices=True)
    h, w = heatmap.shape[-2:]
    coords = torch.arange(h * w, device=heatmap.device).reshape(1, 1, h, w)
    return ixs == coords


def heatmap_to_keypoints(
    heatmap: torch.Tensor, n: int | None = None, window_size: int = 5, score_threshold: float = 0.0
) -> list[Keypoints]:
    """Inference-time nms-based detection protocol."""
    heatmap = heatmap.squeeze(1)
    nms_mask = _nms_mask(heatmap.unsqueeze(1), window_size).squeeze(1)
    nmsed = nms_mask & (heatmap > score_threshold)

    keypoints = []
    for b in range(heatmap.shape[0]):
        yx = nmsed[b].nonzero(as_tuple=False)
        detection_logp = heatmap[b][nmsed[b]]
        xy = yx.flip((1,))

        if n is not None:
            n_ = min(n + 1, detection_logp.numel())
            # torch.kthvalue picks in ascending order and we want to pick in
            # descending order, so we pick n-th smallest among -logp to get
            # -threshold
            minus_threshold, _indices = torch.kthvalue(-detection_logp, n_)
            mask = detection_logp > -minus_threshold

            xy = xy[mask]
            detection_logp = detection_logp[mask]

            # it may be that due to numerical saturation on the threshold we have
            # more than n keypoints, so we need to clip them
            xy = xy[:n]
            detection_logp = detection_logp[:n]

        keypoints.append(Keypoints(xy, detection_logp))

    return keypoints
