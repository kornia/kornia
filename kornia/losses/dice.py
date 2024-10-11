from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR
from kornia.losses._utils import mask_ignore_pixels
from kornia.utils.one_hot import one_hot

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
# https://github.com/Lightning-AI/metrics/blob/v0.11.3/src/torchmetrics/functional/classification/dice.py#L66-L207


def dice_loss(
    pred: Tensor,
    target: Tensor,
    average: str = "micro",
    eps: float = 1e-8,
    weight: Optional[Tensor] = None,
    ignore_index: Optional[int] = -100,
) -> Tensor:
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X \cap Y|}{|X| + |Y|}

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Args:
        pred: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        labels: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C-1`.
        average:
            Reduction applied in multi-class scenario:
            - ``'micro'`` [default]: Calculate the loss across all classes.
            - ``'macro'``: Calculate the loss for each class separately and average the metrics across classes.
        eps: Scalar to enforce numerical stabiliy.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.
        ignore_index: labels with this value are ignored in the loss computation.

    Return:
        One-element tensor of the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = dice_loss(pred, target)
        >>> output.backward()
    """
    KORNIA_CHECK_IS_TENSOR(pred)

    if not len(pred.shape) == 4:
        raise ValueError(f"Invalid pred shape, we expect BxNxHxW. Got: {pred.shape}")

    if not pred.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"pred and target shapes must be the same. Got: {pred.shape} and {target.shape}")

    if not pred.device == target.device:
        raise ValueError(f"pred and target must be in the same device. Got: {pred.device} and {target.device}")
    num_of_classes = pred.shape[1]
    possible_average = {"micro", "macro"}
    KORNIA_CHECK(average in possible_average, f"The `average` has to be one of {possible_average}. Got: {average}")

    # compute softmax over the classes axis
    pred_soft: Tensor = pred.softmax(dim=1)

    target, target_mask = mask_ignore_pixels(target, ignore_index)

    # create the labels one hot tensor
    target_one_hot: Tensor = one_hot(target, num_classes=pred.shape[1], device=pred.device, dtype=pred.dtype)

    # mask ignore pixels
    if target_mask is not None:
        target_mask.unsqueeze_(1)
        target_one_hot = target_one_hot * target_mask
        pred_soft = pred_soft * target_mask

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
    else:
        weight = pred.new_ones(pred.shape[1])

    # set dimensions for the appropriate averaging
    dims: tuple[int, ...] = (2, 3)

    if average == "micro":
        dims = (1, *dims)

        weight = weight.view(-1, 1, 1)
        pred_soft = pred_soft * weight
        target_one_hot = target_one_hot * weight

    intersection = torch.sum(pred_soft * target_one_hot, dims)
    cardinality = torch.sum(pred_soft + target_one_hot, dims)

    dice_score = 2.0 * intersection / (cardinality + eps)
    dice_loss = -dice_score + 1.0

    # reduce the loss across samples (and classes in case of `macro` averaging)
    if average == "macro":
        dice_loss = (dice_loss * weight).sum(-1) / weight.sum()

    dice_loss = torch.mean(dice_loss)

    return dice_loss


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Args:
        average:
            Reduction applied in multi-class scenario:
            - ``'micro'`` [default]: Calculate the loss across all classes.
            - ``'macro'``: Calculate the loss for each class separately and average the metrics across classes.
        eps: Scalar to enforce numerical stabiliy.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.
        ignore_index: labels with this value are ignored in the loss computation.

    Shape:
        - Pred: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C-1`.

    Example:
        >>> N = 5  # num_classes
        >>> criterion = DiceLoss()
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(pred, target)
        >>> output.backward()
    """

    def __init__(
        self,
        average: str = "micro",
        eps: float = 1e-8,
        weight: Optional[Tensor] = None,
        ignore_index: Optional[int] = -100,
    ) -> None:
        super().__init__()
        self.average = average
        self.eps = eps
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return dice_loss(pred, target, self.average, self.eps, self.weight, self.ignore_index)
