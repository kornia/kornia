from __future__ import annotations

import torch

from kornia.core import Tensor
from kornia.utils.misc import eye_like


def create_random_homography(inpt: Tensor, eye_size: int, std_val: float = 1e-3) -> Tensor:
    """Create a batch of random homographies of shape Bx3x3."""
    std = torch.zeros(inpt.shape[0], eye_size, eye_size, device=inpt.device, dtype=inpt.dtype)
    eye = eye_like(eye_size, inpt)
    return eye + std.uniform_(-std_val, std_val)
