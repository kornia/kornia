from __future__ import annotations

import torch

from kornia.core import Tensor


def compute_patch_abs_error(x: Tensor, y: Tensor, h: int, w: int) -> Tensor:
    """Compute the absolute error between patches."""
    return torch.abs(x - y)[..., h // 4 : -h // 4, w // 4 : -w // 4].mean()
