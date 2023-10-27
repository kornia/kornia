from __future__ import annotations

import torch

from kornia.core import Tensor, tensor


def normalize_keypoints(kpts: Tensor, size: Tensor) -> Tensor:
    if isinstance(size, torch.Size):
        size = tensor(size)[None]
    shift = size.float().to(kpts) / 2
    scale = size.max(1).values.float().to(kpts) / 2
    kpts = (kpts - shift[:, None]) / scale[:, None, None]
    return kpts
