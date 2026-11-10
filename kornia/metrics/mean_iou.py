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

import torch

from kornia.core.utils import is_compiling

from .confusion_matrix import confusion_matrix


def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """Compute the IoU of every class in every sample from label maps.

    ``iou = mean_iou(pred, target, num_classes)`` takes class labels of shape ``(B, *)`` and returns float32 IoUs in
    :math:`[0, 1]` with shape ``(B, K)``. Despite the name, nothing is averaged.

    Convention:
        - The labels follow :func:`~kornia.metrics.confusion_matrix`, whose counts the IoU is computed from. To
          average, take the mean over :math:`B` for the mean of per-image IoUs; for the IoU of a whole dataset, sum
          the confusion matrices over all images and take the IoU of the sum.
        - ``eps`` is added to the numerator and the denominator, so a class absent from both maps scores 1 (NaN with
          ``eps=0``), and a class only predicted or only in the target scores about 0. Exclude absent classes before
          averaging over :math:`K`: a larger ``num_classes`` can only raise the average.
          :ref:`Losses and metrics <losses-metrics-conventions>` compares this absent-class rule with the losses,
          :func:`~kornia.metrics.mean_average_precision` and scikit-learn.

    Args:
        pred: predicted class labels in :math:`[0, K)`, with shape ``(B, *)``.
        target: ground-truth class labels in :math:`[0, K)`, with the shape of ``pred``.
        num_classes: the number of classes :math:`K`, at least 2.
        eps: the value added to the numerator and the denominator of every IoU.

    Returns:
        the IoU of every class, with shape ``(B, K)``.

    See Also:
        - :func:`~kornia.metrics.confusion_matrix`: the per-sample counts the IoU is computed from.
        - :func:`~kornia.metrics.mean_iou_bbox`: the IoU of boxes, not label maps.

    Example:
        >>> logits = torch.tensor([[0, 1, 0]])
        >>> target = torch.tensor([[0, 1, 0]])
        >>> mean_iou(logits, target, num_classes=3)
        tensor([[1., 1., 1.]])

    """
    # we first compute the confusion matrix, which validates the labels and the class count
    conf_mat: torch.Tensor = confusion_matrix(pred, target, num_classes)

    # compute the actual intersection over union
    sum_over_row = torch.sum(conf_mat, dim=1)
    sum_over_col = torch.sum(conf_mat, dim=2)
    conf_mat_diag = torch.diagonal(conf_mat, dim1=-2, dim2=-1)
    denominator = sum_over_row + sum_over_col - conf_mat_diag

    # NOTE: we add epsilon so that samples that are neither in the
    # prediction or ground truth are taken into account.
    return (conf_mat_diag + eps) / (denominator + eps)


def _convert_boxes_to_xyxy(boxes: torch.Tensor, box_format: str) -> torch.Tensor:
    """Convert bounding boxes from various formats to xyxy format.

    Args:
        boxes: tensor of bounding boxes in shape (N, 4).
        box_format: box format - one of 'xyxy', 'xywh', or 'cxcywh'.

    Returns:
        boxes in xyxy format (x1, y1, x2, y2).
    """
    if box_format == "xyxy":
        return boxes
    if box_format == "xywh":
        # (x, y, w, h) -> (x1, y1, x2, y2)
        x, y, w, h = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
        x2 = x + w
        y2 = y + h
        return torch.cat([x, y, x2, y2], dim=1)
    if box_format == "cxcywh":
        # (cx, cy, w, h) -> (x1, y1, x2, y2)
        cx, cy, w, h = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.cat([x1, y1, x2, y2], dim=1)
    raise ValueError(f"Unsupported box format: {box_format}. Must be one of 'xyxy', 'xywh', or 'cxcywh'.")


def _promote_integer_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Return integer and bool boxes as float32; floating point boxes are returned unchanged."""
    return boxes if boxes.is_floating_point() else boxes.to(torch.float32)


def mean_iou_bbox(boxes_1: torch.Tensor, boxes_2: torch.Tensor, box_format: str = "xyxy") -> torch.Tensor:
    """Compute the IoU of every pair of boxes from two sets.

    ``iou = mean_iou_bbox(boxes_1, boxes_2)`` takes boxes of shape ``(B1, 4)`` and ``(B2, 4)`` and returns the IoU of
    each box of the first set with each box of the second, as fractions in :math:`[0, 1]`. Despite the name, nothing
    is averaged.

    Convention:
        - Boxes are exclusive in every ``box_format``: the area of ``(x1, y1, x2, y2)`` is
          :math:`(x_2 - x_1)(y_2 - y_1)`, with no ``+ 1``, as in :func:`~kornia.geometry.bbox.nms`. A box with a
          non-positive width or height raises ``AssertionError``, unless the call captures a graph
          (``torch.compile`` or ``torch.export``), which skips the check and gives such a box an IoU of 0 or NaN. A
          :class:`~kornia.geometry.boxes.Boxes` gives the same IoU through ``to_tensor('xyxy')``, not through its
          inclusive ``'xyxy_plus'`` export.
        - Swapping the two sets transposes the result.

    Args:
        boxes_1: the first set of boxes, with shape ``(B1, 4)``.
        boxes_2: the second set of boxes, with shape ``(B2, 4)``.
        box_format: ``'xyxy'`` for ``(x1, y1, x2, y2)`` with the top-left and bottom-right corners, ``'xywh'`` for
            ``(x, y, w, h)`` with the top-left corner, or ``'cxcywh'`` for ``(cx, cy, w, h)`` with the center.

    Returns:
        the IoU matrix with shape ``(B1, B2)``.

    .. note::
        Integer (and bool) boxes are computed in ``float32``, so the result is ``float32`` and widths, areas and the
        union do not wrap around in a narrow integer dtype such as ``uint8``, ``int8`` or ``int16``.

    See Also:
        - :func:`~kornia.metrics.mean_iou`: the IoU of label maps, not boxes.
        - :func:`~kornia.metrics.mean_average_precision`: scores detections with this IoU.

    Example:
        >>> # XYXY format
        >>> boxes_1 = torch.tensor([[40, 40, 60, 60], [30, 40, 50, 60]])
        >>> boxes_2 = torch.tensor([[40, 50, 60, 70], [30, 40, 40, 50]])
        >>> mean_iou_bbox(boxes_1, boxes_2)
        tensor([[0.3333, 0.0000],
                [0.1429, 0.2500]])
        >>> # XYWH format
        >>> boxes_1_xywh = torch.tensor([[40, 40, 20, 20], [30, 40, 20, 20]])
        >>> boxes_2_xywh = torch.tensor([[40, 50, 20, 20], [30, 40, 10, 10]])
        >>> mean_iou_bbox(boxes_1_xywh, boxes_2_xywh, box_format='xywh')
        tensor([[0.3333, 0.0000],
                [0.1429, 0.2500]])
        >>> # CXCYWH format
        >>> boxes_1_cxcywh = torch.tensor([[50, 50, 20, 20], [40, 50, 20, 20]])
        >>> boxes_2_cxcywh = torch.tensor([[50, 60, 20, 20], [35, 45, 10, 10]])
        >>> mean_iou_bbox(boxes_1_cxcywh, boxes_2_cxcywh, box_format='cxcywh')
        tensor([[0.3333, 0.0000],
                [0.1429, 0.2500]])

    """
    # Integer boxes would otherwise form every width, area and the union in their own dtype, where uint8, int8 and
    # int16 wrap around. Promote them before the format conversion too, because 'xywh' adds x + w in the box dtype.
    # Bool boxes are covered as well (they cannot be subtracted). The result is float32 for integer input either way.
    boxes_1 = _promote_integer_boxes(boxes_1)
    boxes_2 = _promote_integer_boxes(boxes_2)

    output_dtype = torch.promote_types(boxes_1.dtype, boxes_2.dtype)
    # Compute half-precision boxes in float32 from the start: the corners (x + w) round in their dtype, which can erase
    # a small width at a large position, and image-sized areas overflow float16.
    if boxes_1.dtype in (torch.float16, torch.bfloat16):
        boxes_1 = boxes_1.float()
    if boxes_2.dtype in (torch.float16, torch.bfloat16):
        boxes_2 = boxes_2.float()

    # Convert boxes to xyxy format
    boxes_1_xyxy = _convert_boxes_to_xyxy(boxes_1, box_format)
    boxes_2_xyxy = _convert_boxes_to_xyxy(boxes_2, box_format)

    # Validate boxes are in proper xyxy format. The checks read the data, which graph capture cannot do;
    # skip them under any capture. TorchScript cannot call is_compiling(), and a scripted call keeps the checks.
    if torch.jit.is_scripting() or not is_compiling():
        if not (
            ((boxes_1_xyxy[:, 2] - boxes_1_xyxy[:, 0]) > 0).all()
            and ((boxes_1_xyxy[:, 3] - boxes_1_xyxy[:, 1]) > 0).all()
        ):
            raise AssertionError("Boxes_1 contains invalid boxes after conversion.")
        if not (
            ((boxes_2_xyxy[:, 2] - boxes_2_xyxy[:, 0]) > 0).all()
            and ((boxes_2_xyxy[:, 3] - boxes_2_xyxy[:, 1]) > 0).all()
        ):
            raise AssertionError("Boxes_2 contains invalid boxes after conversion.")

    # Find intersection
    lower_bounds = torch.max(boxes_1_xyxy[:, :2].unsqueeze(1), boxes_2_xyxy[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(boxes_1_xyxy[:, 2:].unsqueeze(1), boxes_2_xyxy[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    intersection = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (boxes_1_xyxy[:, 2] - boxes_1_xyxy[:, 0]) * (boxes_1_xyxy[:, 3] - boxes_1_xyxy[:, 1])  # (n1)
    areas_set_2 = (boxes_2_xyxy[:, 2] - boxes_2_xyxy[:, 0]) * (boxes_2_xyxy[:, 3] - boxes_2_xyxy[:, 1])  # (n2)

    # Find the union
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    iou = intersection / union  # (n1, n2)
    return iou.to(output_dtype) if output_dtype in (torch.float16, torch.bfloat16) else iou
