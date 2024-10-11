from __future__ import annotations

from typing import Optional

import torch


def mask_ignore_pixels(
    target: torch.Tensor, ignore_index: Optional[int]
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if ignore_index is None:
        return target, None

    target_mask = target != ignore_index

    if target_mask.all():
        return target, None

    # map invalid pixels to a valid class (0)
    # they need to be manually excluded from the loss computation after
    target = target.where(target_mask, target.new_zeros(1))

    return target, target_mask
