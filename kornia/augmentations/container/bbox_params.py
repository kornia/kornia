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

"""BboxParams — Albumentations-style bbox handling for AugmentationSequential.

When passed via ``bbox_params=`` to ``AugmentationSequential``, every batch
gets:
- Boxes clamped to image bounds after geometric transforms
- Boxes with area < min_area or visibility < min_visibility removed
- Corresponding label_fields entries removed in lockstep

This eliminates the boilerplate downstream users (rf-detr, torchgeo) had to
write themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor


@dataclass
class BboxParams:
    """Configuration for bbox handling in AugmentationSequential.

    Args:
        format: Bbox layout. One of ``"xyxy"``, ``"xywh"``, ``"cxcywh"``.
        min_area: Drop boxes whose post-augmentation area falls below this many
            pixels squared. Default 0 (no minimum).
        min_visibility: Drop boxes whose post/pre area ratio falls below this
            fraction. Range [0.0, 1.0]. Default 0.0 (no minimum).
        label_fields: Names of target tensors that should be filtered in lockstep
            with the boxes. Each field is expected to be a 1-D tensor with the
            same first-axis length as the boxes.
    """

    format: str = "xyxy"

    min_area: float = 0.0
    min_visibility: float = 0.0
    label_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.format not in ("xyxy", "xywh", "cxcywh"):
            raise ValueError(f"format must be one of xyxy/xywh/cxcywh; got {self.format!r}")
        if not (0.0 <= self.min_visibility <= 1.0):
            raise ValueError(f"min_visibility must be in [0,1]; got {self.min_visibility}")
        if self.min_area < 0:
            raise ValueError(f"min_area must be >= 0; got {self.min_area}")


def _bbox_area_xyxy(boxes: Tensor) -> Tensor:
    """Compute area assuming xyxy format. ``boxes`` shape: (..., 4)."""
    w = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
    h = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    return w * h


def _to_xyxy(boxes: Tensor, fmt: str) -> Tensor:
    if fmt == "xyxy":
        return boxes
    if fmt == "xywh":
        x, y, w, h = boxes.unbind(-1)
        return torch.stack([x, y, x + w, y + h], dim=-1)
    if fmt == "cxcywh":
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    raise ValueError(f"unknown format {fmt!r}")


def _from_xyxy(boxes: Tensor, fmt: str) -> Tensor:
    if fmt == "xyxy":
        return boxes
    if fmt == "xywh":
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)
    if fmt == "cxcywh":
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)
    raise ValueError(f"unknown format {fmt!r}")


def filter_bboxes(
    boxes_pre: Tensor,
    boxes_post: Tensor,
    image_size: tuple[int, int],
    params: BboxParams,
    labels: dict[str, Tensor] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply clamp + min_area + min_visibility filtering.

    Args:
        boxes_pre: Boxes before augmentation, shape ``(B, N, 4)`` or ``(N, 4)``.
        boxes_post: Boxes after augmentation, same shape.
        image_size: ``(H, W)``.
        params: BboxParams config.
        labels: Optional dict mapping label_field name -> tensor with first
            dim == N.

    Returns:
        Filtered boxes (same shape as input, padded with zeros for dropped
        entries) and a dict ``{name: filtered_tensor}`` for each label field.
        For now, the simple per-batch implementation: returns the filtered
        sub-tensor when input is 2-D ``(N, 4)``; raises NotImplementedError
        for 3-D batched input (caller should iterate).
    """
    H, W = image_size
    fmt = params.format
    pre_xyxy = _to_xyxy(boxes_pre, fmt)
    post_xyxy = _to_xyxy(boxes_post, fmt)

    # Clamp post to image bounds
    post_clamped = post_xyxy.clone()
    post_clamped[..., 0::2] = post_clamped[..., 0::2].clamp(min=0, max=W)
    post_clamped[..., 1::2] = post_clamped[..., 1::2].clamp(min=0, max=H)

    pre_area = _bbox_area_xyxy(pre_xyxy)
    post_area = _bbox_area_xyxy(post_clamped)

    keep = post_area >= params.min_area
    if params.min_visibility > 0:
        # Avoid div-by-zero
        vis = post_area / pre_area.clamp(min=1e-6)
        keep = keep & (vis >= params.min_visibility)

    if post_clamped.dim() == 3:
        raise NotImplementedError("batched filter_bboxes not yet implemented; caller should iterate")

    out_boxes_xyxy = post_clamped[keep]
    out_boxes = _from_xyxy(out_boxes_xyxy, fmt)
    out_labels: dict[str, Tensor] = {}
    if labels:
        for name in params.label_fields:
            if name in labels:
                out_labels[name] = labels[name][keep]
    return out_boxes, out_labels


__all__ = ["BboxParams", "filter_bboxes"]
