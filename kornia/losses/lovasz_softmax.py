from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from kornia.testing import KORNIA_CHECK_SHAPE

# based on:
# https://github.com/bermanmaxim/LovaszSoftmax


def lovasz_softmax_loss(pred: Tensor, target: Tensor) -> Tensor:
    r"""Criterion that computes a surrogate multi-class intersection-over-union (IoU) loss.

    According to [1], we compute the IoU as follows:

    .. math::

        \text{IoU}(x, class) = \frac{|X \cap Y|}{|X \cup Y|}

    [1] approximates this fomular with a surrogate, which is fully differentable.

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the binary tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{IoU}(x, class)

    Reference:
        [1] https://arxiv.org/pdf/1705.08790.pdf

    . note::
        This loss function only supports multi-class (C > 1) labels. For binary
        labels please use the Lovasz-Hinge loss.

    Args:
        pred: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes > 1.
        labels: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`.

    Return:
        a scalar with the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = lovasz_softmax_loss(pred, target)
        >>> output.backward()
    """
    KORNIA_CHECK_SHAPE(pred, ["B", "N", "H", "W"])

    KORNIA_CHECK_SHAPE(target, ["B", "H", "W"])

    if not pred.shape[1] > 1:
        raise ValueError(f"Invalid pred shape, we expect BxNxHxW, with N > 1. Got: {pred.shape}")

    if not pred.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"pred and target shapes must be the same. Got: {pred.shape} and {target.shape}")

    if not pred.device == target.device:
        raise ValueError(f"pred and target must be in the same device. Got: {pred.device} and {target.device}")

    # flatten pred [B, C, -1] and target [B, -1] and to float
    pred_flatten: Tensor = pred.reshape(pred.shape[0], pred.shape[1], -1)
    target_flatten: Tensor = target.reshape(target.shape[0], -1).float()

    # get shapes
    B, C, N = pred_flatten.shape

    # compute softmax over the classes axis
    pred_soft: Tensor = pred_flatten.softmax(1)

    # compute actual loss
    losses: list[Tensor] = []
    batch_index: Tensor = torch.arange(B, device=pred.device).reshape(-1, 1).repeat(1, N).reshape(-1)
    for c in range(C):
        foreground: Tensor = 1.0 * (target_flatten == c)
        class_pred: Tensor = pred_soft[:, c]
        errors = (class_pred - foreground).abs()
        errors_sorted, permutation = torch.sort(errors, dim=1, descending=True)
        target_sorted: Tensor = target_flatten[batch_index, permutation.view(-1)]
        target_sorted = target_sorted.view(B, N)
        target_sorted_sum: Tensor = target_sorted.sum(1, keepdim=True)
        intersection: Tensor = target_sorted_sum - target_sorted.cumsum(1)
        union: Tensor = target_sorted_sum + (1.0 - target_sorted).cumsum(1)
        gradient: Tensor = 1.0 - intersection / union
        if N > 1:
            gradient[..., 1:] = gradient[..., 1:] - gradient[..., :-1]
        loss: Tensor = (errors_sorted.relu() * gradient).sum(1).mean()
        losses.append(loss)
    final_loss: Tensor = torch.stack(losses, dim=0).mean()
    return final_loss


class LovaszSoftmaxLoss(nn.Module):
    r"""Criterion that computes a surrogate multi-class intersection-over-union (IoU) loss.

    According to [1], we compute the IoU as follows:

    .. math::

        \text{IoU}(x, class) = \frac{|X \cap Y|}{|X \cup Y|}

    [1] approximates this fomular with a surrogate, which is fully differentable.

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the binary tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{IoU}(x, class)

    Reference:
        [1] https://arxiv.org/pdf/1705.08790.pdf

    . note::
        This loss function only supports multi-class (C > 1) labels. For binary
        labels please use the Lovasz-Hinge loss.

    Args:
        pred: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes > 1.
        labels: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`.

    Return:
        a scalar with the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> criterion = LovaszSoftmaxLoss()
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(pred, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return lovasz_softmax_loss(pred=pred, target=target)
