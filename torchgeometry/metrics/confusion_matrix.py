from typing import Optional

import torch


def confusion_matrix(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        num_classes: int) -> torch.Tensor:
    r"""Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true (torch.Tensor) : tensor with ground truth (correct) target
          values. The shape can be :math:`(N, *)`.
        y_pred (torch.Tensor) : tensor with estimated targets returned by a
          classifier. The shape can be :math:`(N, *)`.
        num_classes (int): total number of classes in y_true.

    Returns:
        torch.Tensor: a tensor with the confusion matrix with shape
        :math:`(N, C, C)` where C is the number of classes.
    """
    if not torch.is_tensor(y_true):
        raise TypeError("Input y_true type is not a torch.Tensor. Got {}"
                        .format(type(y_true)))
    if not torch.is_tensor(y_pred):
        raise TypeError("Input y_pred type is not a torch.Tensor. Got {}"
                        .format(type(y_pred)))
    if not y_true.shape == y_pred.shape:
        raise ValueError("Inputs y_true and y_pred must have the same shape. "
                         "Got: {}".format(y_true.shape, y_pred.shape))
    if not y_true.device == y_pred.device:
        raise ValueError("Inputs must be in the same device. "
                         "Got: {} - {}".format(y_true.device, y_pred.device))
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("The number of classes must be an intenger bigger "
                         "than two. Got: {}".format(num_classes))
    batch_size: int = y_true.shape[0]
    y_true_vec: torch.Tensor = y_true.view(batch_size, -1)
    y_pred_vec: torch.Tensor = y_pred.view(batch_size, -1)

    # NOTE: torch.bincount does not implement batched version
    pre_bincount: torch.Tensor = y_true_vec * num_classes + y_pred_vec
    confusion_vec: torch.Tensor = torch.stack([
        torch.bincount(pb) for pb in pre_bincount
    ])
    return confusion_vec.view(batch_size, num_classes, num_classes)  # BxNxN
