from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

# based on:
# https://github.com/bermanmaxim/LovaszSoftmax


def lovasz_softmax_loss(pred: Tensor, target: Tensor, weight: Optional[Tensor] = None) -> Tensor:
    r"""Criterion that computes a surrogate multi-class intersection-over-union (IoU) loss.

    According to [1], we compute the IoU as follows:

    .. math::

        \text{IoU}(x, class) = \frac{|X \cap Y|}{|X \cup Y|}

    [1] approximates this fomular with a surrogate, which is fully differentable.

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the long tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{IoU}(x, class)

    Reference:
        [1] https://arxiv.org/pdf/1705.08790.pdf

    .. note::
        This loss function only supports multi-class (C > 1) labels. For binary
        labels please use the Lovasz-Hinge loss.

    Args:
        pred: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes > 1.
        labels: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C-1`.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.

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

    num_of_classes = pred.shape[1]
    # compute the actual dice score
    if weight is not None:
        KORNIA_CHECK_IS_TENSOR(weight, "weight must be Tensor or None.")
        KORNIA_CHECK(
            (weight.shape[0] == num_of_classes and weight.numel() == num_of_classes),
            f"weight shape must be (num_of_classes,): ({num_of_classes},), got {weight.shape}",
        )
        KORNIA_CHECK(
            weight.device == pred.device,
            f"weight and pred must be in the same device. Got: {weight.device} and {pred.device}",
        )

    # flatten pred [B, C, -1] and target [B, -1] and to float
    pred_flatten: Tensor = pred.reshape(pred.shape[0], pred.shape[1], -1)
    target_flatten: Tensor = target.reshape(target.shape[0], -1)

    # get shapes
    B, C, N = pred_flatten.shape

    # compute softmax over the classes axis
    pred_soft: Tensor = pred_flatten.softmax(1)

    # compute actual loss
    foreground: Tensor = (
        torch.nn.functional.one_hot(target_flatten.to(torch.int64), num_classes=C).permute(0, 2, 1).to(pred.dtype)
    )
    errors: Tensor = (pred_soft - foreground).abs()
    errors_sorted, permutations = torch.sort(errors, dim=2, descending=True)
    batch_index = torch.arange(B, device=pred.device).unsqueeze(1).unsqueeze(2).expand(B, C, N)
    target_sorted = target_flatten[batch_index, permutations]
    target_sorted_sum = target_sorted.sum(2, keepdim=True)
    intersection = target_sorted_sum - target_sorted.cumsum(2)
    union = target_sorted_sum + (1.0 - target_sorted).cumsum(2)
    gradient = 1.0 - intersection / union
    if N > 1:
        gradient[..., 1:] = gradient[..., 1:] - gradient[..., :-1]
    weighted_errors = errors_sorted * gradient
    loss_per_class = weighted_errors.sum(2).mean(0)
    if weight is not None:
        loss_per_class *= weight
    final_loss: Tensor = loss_per_class.mean()
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

    .. note::
        This loss function only supports multi-class (C > 1) labels. For binary
        labels please use the Lovasz-Hinge loss.

    Args:
        pred: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes > 1.
        labels: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C-1`.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.

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

    def __init__(self, weight: Optional[Tensor] = None) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return lovasz_softmax_loss(pred=pred, target=target, weight=self.weight)
