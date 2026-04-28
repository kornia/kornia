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

"""BboxParams dataclass and filter_bboxes utility for Albumentations-style bounding-box filtering."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

__all__ = ["BboxParams", "filter_bboxes"]

_VALID_FORMATS = frozenset({"xyxy", "xywh", "cxcywh"})


@dataclass(frozen=True)
class BboxParams:
    """Configuration for bounding-box format and filtering thresholds.

    Args:
        format: Bounding-box coordinate format. One of ``"xyxy"``, ``"xywh"``,
            or ``"cxcywh"``.
        min_area: Minimum post-transform box area (in pixels squared) to keep.
            Must be >= 0.
        min_visibility: Minimum fraction of the original box area that must
            remain after the transform to keep the box. Must be in [0.0, 1.0].
        label_fields: Names of label tensors that should be filtered in
            lock-step with the bounding boxes.

    Example:
        >>> params = BboxParams(format="xyxy", min_area=100.0, min_visibility=0.5)
        >>> params.format
        'xyxy'
    """

    format: str = "xyxy"
    min_area: float = 0.0
    min_visibility: float = 0.0
    label_fields: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.format not in _VALID_FORMATS:
            raise ValueError(f"BboxParams.format must be one of {sorted(_VALID_FORMATS)!r}, got {self.format!r}.")
        if not (0.0 <= self.min_visibility <= 1.0):
            raise ValueError(f"BboxParams.min_visibility must be in [0.0, 1.0], got {self.min_visibility}.")
        if self.min_area < 0.0:
            raise ValueError(f"BboxParams.min_area must be >= 0, got {self.min_area}.")


# ---------------------------------------------------------------------------
# Internal coordinate helpers
# ---------------------------------------------------------------------------


def _to_xyxy(boxes: Tensor, fmt: str) -> Tensor:
    """Convert (N, 4) boxes from *fmt* to xyxy."""
    if fmt == "xyxy":
        return boxes.clone()
    if fmt == "xywh":
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1)
    if fmt == "cxcywh":
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        half_w = boxes[:, 2] / 2.0
        half_h = boxes[:, 3] / 2.0
        return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=1)
    raise ValueError(f"Unknown format: {fmt!r}")  # pragma: no cover


def _from_xyxy(boxes: Tensor, fmt: str) -> Tensor:
    """Convert (N, 4) boxes from xyxy back to *fmt*."""
    if fmt == "xyxy":
        return boxes.clone()
    if fmt == "xywh":
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=1)
    if fmt == "cxcywh":
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = x2 - x1
        h = y2 - y1
        return torch.stack([x1 + w / 2.0, y1 + h / 2.0, w, h], dim=1)
    raise ValueError(f"Unknown format: {fmt!r}")  # pragma: no cover


def _box_area(boxes_xyxy: Tensor) -> Tensor:
    """Return per-box area clamped to >= 0, shape (N,)."""
    w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0.0)
    h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0.0)
    return w * h


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_bboxes(
    boxes_pre: Tensor,
    boxes_post: Tensor,
    image_size: tuple[int, int],
    params: BboxParams,
    labels: dict[str, Tensor] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Filter bounding boxes after an augmentation transform.

    Boxes that fall below ``params.min_area`` or ``params.min_visibility``
    after the transform are removed.  The corresponding rows in every
    ``label_fields`` tensor are removed in lock-step.

    Args:
        boxes_pre: Pre-transform boxes, shape ``(N, 4)``, in the coordinate
            format given by ``params.format``.
        boxes_post: Post-transform boxes, shape ``(N, 4)``, in the same
            format.
        image_size: ``(height, width)`` of the output image, used to clamp
            post-transform boxes to valid image bounds.
        params: :class:`BboxParams` configuration object.
        labels: Optional mapping of label-field name to 1-D tensor of length
            ``N``.  Only fields listed in ``params.label_fields`` are returned
            in the output dict; extra keys are silently ignored.

    Returns:
        A tuple ``(filtered_boxes, filtered_labels)`` where:

        - ``filtered_boxes`` is shape ``(M, 4)`` in ``params.format``.
        - ``filtered_labels`` is a ``dict[str, Tensor]`` with the kept rows
          for every name in ``params.label_fields``.

    Raises:
        NotImplementedError: If ``boxes_pre`` or ``boxes_post`` is 3-D
            (batched).

    Example:
        >>> import torch
        >>> from kornia.augmentation.container.bbox_params import BboxParams, filter_bboxes
        >>> params = BboxParams(format="xyxy", min_area=0.0, min_visibility=0.0)
        >>> boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        >>> out, _ = filter_bboxes(boxes, boxes, (100, 100), params)
        >>> out.shape
        torch.Size([1, 4])
    """
    if boxes_pre.ndim == 3 or boxes_post.ndim == 3:
        raise NotImplementedError(
            "filter_bboxes does not yet support batched (3-D) boxes. Pass 2-D (N, 4) tensors instead."
        )

    if labels is None:
        labels = {}

    h, w = image_size
    fmt = params.format

    # Work in xyxy internally.
    pre_xyxy = _to_xyxy(boxes_pre, fmt)
    post_xyxy = _to_xyxy(boxes_post, fmt)

    # Clamp post boxes to image bounds.
    post_xyxy_clamped = post_xyxy.clone()
    post_xyxy_clamped[:, 0] = post_xyxy_clamped[:, 0].clamp(0.0, float(w))
    post_xyxy_clamped[:, 1] = post_xyxy_clamped[:, 1].clamp(0.0, float(h))
    post_xyxy_clamped[:, 2] = post_xyxy_clamped[:, 2].clamp(0.0, float(w))
    post_xyxy_clamped[:, 3] = post_xyxy_clamped[:, 3].clamp(0.0, float(h))

    pre_area = _box_area(pre_xyxy)
    post_area = _box_area(post_xyxy_clamped)

    # Build keep mask.
    keep = post_area >= params.min_area

    if params.min_visibility > 0.0:
        # Avoid division by zero: boxes with zero pre_area are always dropped.
        safe_pre = pre_area.clamp(min=1e-9)
        visibility = post_area / safe_pre
        keep = keep & (visibility >= params.min_visibility)

    # Filter boxes and convert back to original format.
    filtered_xyxy = post_xyxy_clamped[keep]
    filtered_boxes = _from_xyxy(filtered_xyxy, fmt)

    # Filter label fields.
    filtered_labels: dict[str, Tensor] = {}
    for name in params.label_fields:
        if name in labels:
            filtered_labels[name] = labels[name][keep]

    return filtered_boxes, filtered_labels
