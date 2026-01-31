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

from .confusion_matrix import confusion_matrix


def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    r"""Calculate mean Intersection-Over-Union (mIOU).

    The function internally computes the confusion matrix.

    Args:
        pred : tensor with estimated targets returned by a
          classifier. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1.
        target: tensor with ground truth (correct) target
          values. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1, where targets are assumed to be provided as
          one-hot vectors.
        num_classes: total possible number of classes in target.
        eps: epsilon for numerical stability.

    Returns:
        a tensor representing the mean intersection-over union
        with shape :math:`(B, K)` where K is the number of classes.

    Example:
        >>> logits = torch.tensor([[0, 1, 0]])
        >>> target = torch.tensor([[0, 1, 0]])
        >>> mean_iou(logits, target, num_classes=3)
        tensor([[1., 1., 1.]])

    """
    if not torch.is_tensor(pred) and pred.dtype is not torch.int64:
        raise TypeError(f"Input pred type is not a torch.Tensor with torch.int64 dtype. Got {type(pred)}")

    if not torch.is_tensor(target) and target.dtype is not torch.int64:
        raise TypeError(f"Input target type is not a torch.Tensor with torch.int64 dtype. Got {type(target)}")
    if not pred.shape == target.shape:
        raise ValueError(f"Inputs pred and target must have the same shape. Got: {pred.shape} and {target.shape}")
    if not pred.device == target.device:
        raise ValueError(f"Inputs must be in the same device. Got: {pred.device} - {target.device}")

    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"The number of classes must be an integer bigger than two. Got: {num_classes}")

    # we first compute the confusion matrix
    conf_mat: torch.Tensor = confusion_matrix(pred, target, num_classes)

    # compute the actual intersection over union
    sum_over_row = torch.sum(conf_mat, dim=1)
    sum_over_col = torch.sum(conf_mat, dim=2)
    conf_mat_diag = torch.diagonal(conf_mat, dim1=-2, dim2=-1)
    denominator = sum_over_row + sum_over_col - conf_mat_diag

    # NOTE: we add epsilon so that samples that are neither in the
    # prediction or ground truth are taken into account.
    ious = (conf_mat_diag + eps) / (denominator + eps)
    return ious

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
    elif box_format == "xywh":
        # (x, y, w, h) -> (x1, y1, x2, y2)
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x2 = x + w
        y2 = y + h
        return torch.stack([x, y, x2, y2], dim=1)
    elif box_format == "cxcywh":
        # (cx, cy, w, h) -> (x1, y1, x2, y2)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    else:
        raise ValueError(f"Unsupported box format: {box_format}. Must be one of 'xyxy', 'xywh', or 'cxcywh'.")


def mean_iou_bbox(boxes_1: torch.Tensor, boxes_2: torch.Tensor, box_format: str = "xyxy") -> torch.Tensor:
    """Compute the IoU of the cartesian product of two sets of boxes.

    Args:
        boxes_1: a tensor of bounding boxes in :math:`(B1, 4)`.
        boxes_2: a tensor of bounding boxes in :math:`(B2, 4)`.
        box_format: the bounding box format. Supported formats are:
            - 'xyxy': (x1, y1, x2, y2) where (x1, y1) is top-left and (x2, y2) is bottom-right
            - 'xywh': (x, y, w, h) where (x, y) is top-left, w is width, h is height
            - 'cxcywh': (cx, cy, w, h) where (cx, cy) is center, w is width, h is height
            Default: 'xyxy'.

    Returns:
        a tensor in dimensions :math:`(B1, B2)`, representing the
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2.

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
    # Convert boxes to xyxy format
    boxes_1_xyxy = _convert_boxes_to_xyxy(boxes_1, box_format)
    boxes_2_xyxy = _convert_boxes_to_xyxy(boxes_2, box_format)
    
    # Validate boxes are in proper xyxy format
    if not (((boxes_1_xyxy[:, 2] - boxes_1_xyxy[:, 0]) > 0).all() and ((boxes_1_xyxy[:, 3] - boxes_1_xyxy[:, 1]) > 0).all()):
        raise AssertionError("Boxes_1 contains invalid boxes after conversion.")
    if not (((boxes_2_xyxy[:, 2] - boxes_2_xyxy[:, 0]) > 0).all() and ((boxes_2_xyxy[:, 3] - boxes_2_xyxy[:, 1]) > 0).all()):
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

    return intersection / union  # (n1, n2)