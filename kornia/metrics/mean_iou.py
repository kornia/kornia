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


def mean_iou_bbox(boxes_1: torch.Tensor, boxes_2: torch.Tensor) -> torch.Tensor:
    """Compute the IoU of the cartesian product of two sets of boxes.

    Each box in each set shall be (x1, y1, x2, y2).

    Args:
        boxes_1: a tensor of bounding boxes in :math:`(B1, 4)`.
        boxes_2: a tensor of bounding boxes in :math:`(B2, 4)`.

    Returns:
        a tensor in dimensions :math:`(B1, B2)`, representing the
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2.

    Example:
        >>> boxes_1 = torch.tensor([[40, 40, 60, 60], [30, 40, 50, 60]])
        >>> boxes_2 = torch.tensor([[40, 50, 60, 70], [30, 40, 40, 50]])
        >>> mean_iou_bbox(boxes_1, boxes_2)
        tensor([[0.3333, 0.0000],
                [0.1429, 0.2500]])
    """
    # TODO: support more box types. e.g. xywh,
    if not (((boxes_1[:, 2] - boxes_1[:, 0]) > 0).all() or ((boxes_1[:, 3] - boxes_1[:, 1]) > 0).all()):
        raise AssertionError("Boxes_1 does not follow (x1, y1, x2, y2) format.")
    if not (((boxes_2[:, 2] - boxes_2[:, 0]) > 0).all() or ((boxes_2[:, 3] - boxes_2[:, 1]) > 0).all()):
        raise AssertionError("Boxes_2 does not follow (x1, y1, x2, y2) format.")
    # find intersection
    lower_bounds = torch.max(boxes_1[:, :2].unsqueeze(1), boxes_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(boxes_1[:, 2:].unsqueeze(1), boxes_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    intersection = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (boxes_1[:, 2] - boxes_1[:, 0]) * (boxes_1[:, 3] - boxes_1[:, 1])  # (n1)
    areas_set_2 = (boxes_2[:, 2] - boxes_2[:, 0]) * (boxes_2[:, 3] - boxes_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)
