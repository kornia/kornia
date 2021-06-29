from typing import Optional

import torch

from kornia.utils.metrics.confusion_matrix import confusion_matrix


def mean_iou(input: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    r"""Calculate mean Intersection-Over-Union (mIOU).

    The function internally computes the confusion matrix.

    Args:
        input : tensor with estimated targets returned by a
          classifier. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1.
        target: tensor with ground truth (correct) target
          values. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1, where targets are assumed to be provided as
          one-hot vectors.
        num_classes: total possible number of classes in target.

    Returns:
        ta tensor representing the mean intersection-over union
        with shape :math:`(B, K)` where K is the number of classes.
    """
    if not torch.is_tensor(input) and input.dtype is not torch.int64:
        raise TypeError("Input input type is not a torch.Tensor with " "torch.int64 dtype. Got {}".format(type(input)))

    if not torch.is_tensor(target) and target.dtype is not torch.int64:
        raise TypeError(
            "Input target type is not a torch.Tensor with " "torch.int64 dtype. Got {}".format(type(target))
        )
    if not input.shape == target.shape:
        raise ValueError(
            "Inputs input and target must have the same shape. " "Got: {} and {}".format(input.shape, target.shape)
        )
    if not input.device == target.device:
        raise ValueError("Inputs must be in the same device. " "Got: {} - {}".format(input.device, target.device))

    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("The number of classes must be an integer bigger " "than two. Got: {}".format(num_classes))

    # we first compute the confusion matrix
    conf_mat: torch.Tensor = confusion_matrix(input, target, num_classes)

    # compute the actual intersection over union
    sum_over_row = torch.sum(conf_mat, dim=1)
    sum_over_col = torch.sum(conf_mat, dim=2)
    conf_mat_diag = torch.diagonal(conf_mat, dim1=-2, dim2=-1)
    denominator = sum_over_row + sum_over_col - conf_mat_diag

    # NOTE: we add epsilon so that samples that are neither in the
    # prediction or ground truth are taken into account.
    ious = (conf_mat_diag + eps) / (denominator + eps)
    return ious
