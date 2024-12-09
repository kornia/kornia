from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from kornia.core import Tensor

from .structs import Keypoints


def nms(signal: Tensor, window_size: int = 5, cutoff: float = 0.0) -> Tensor:
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
    heatmap = heatmap.squeeze(1)
    nmsed = nms(heatmap, window_size=window_size, cutoff=score_threshold)

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
