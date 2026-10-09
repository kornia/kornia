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

"""Module containing the Sørensen-Dice Coefficient loss."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR
from kornia.losses._utils import mask_ignore_pixels
from kornia.losses.one_hot import one_hot

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
# https://github.com/Lightning-AI/metrics/blob/v0.11.3/src/torchmetrics/functional/classification/dice.py#L66-L207


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    average: str = "micro",
    eps: float = 1e-8,
    weight: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = -100,
) -> torch.Tensor:
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X \cap Y|}{|X| + |Y|}

    Where:
       - :math:`X` is the softmax of ``pred`` over the classes.
       - :math:`Y` is the one-hot encoding of ``target`` by :func:`~kornia.losses.one_hot`.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Convention:
        - ``pred`` and ``target`` are those of :func:`~kornia.losses.focal_loss` (logits with the softmax taken
          inside, int64 class indices; see its Convention block), restricted to ``(B, C, H, W)`` and ``(B, H, W)``.
          A one-channel ``pred`` has a softmax of 1 everywhere and gives a constant loss with zero gradient: a binary
          task passes two channels.
        - ``'micro'`` (default) computes one Dice per sample over all classes and pixels, ``weight`` scaling the
          terms of class ``c`` in both sums: :math:`1 - 2 \sum_c w_c |X_c \cap Y_c| / \sum_c w_c (|X_c| + |Y_c|)`.
          ``'macro'`` computes one Dice per sample and class and averages them as
          :math:`\sum_c w_c \, \text{loss}_c / \sum_c w_c` over the classes present in the sample's non-ignored
          target: a class that is predicted but absent from the target is left out. Both then average over the batch
          and return a 0-d tensor; there is no ``reduction``.
        - ``eps`` is added to the denominator only: :math:`1 - 2 |X \cap Y| / (|X| + |Y| + \epsilon)`.
        - Pixels labelled ``ignore_index`` (default ``-100``) leave both sums of their own sample. A sample whose
          pixels are all ignored, or whose present classes all have weight 0, enters the batch mean as loss 1 with a
          zero gradient, at ``eps=0`` as well.

    Args:
        pred: logits torch.Tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        target: labels torch.Tensor with shape :math:`(N, H, W)` where each value
          is in range :math:`0 ≤ targets[i] ≤ C-1`.
        average:
            Reduction applied in multi-class scenario:

            - ``'micro'`` [default]: Calculate the loss across all classes.
            - ``'macro'``: Average the class losses of each sample; see the Convention block.
        eps: Scalar added to the denominator of the Dice ratio for numerical stability.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`; see the Convention block for how each
          ``average`` applies them.
        ignore_index: labels with this value are ignored in the loss computation.

    Return:
        One-element torch.Tensor of the computed loss.

    Note:
        Spatial reductions use float32 for float16 and bfloat16 inputs to avoid
        overflow on large images. The returned loss retains the usual dtype
        promotion between the inputs and class weights.

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

    if not (pred.shape[0] == target.shape[0] and pred.shape[2:] == target.shape[1:]):
        raise ValueError(f"Expected target size {torch.Size((pred.shape[0], *pred.shape[2:]))}, got {target.shape}")
    num_of_classes = pred.shape[1]
    possible_average = {"micro", "macro"}
    KORNIA_CHECK(average in possible_average, f"The `average` has to be one of {possible_average}. Got: {average}")

    # compute F.softmax over the classes axis
    pred_soft: torch.Tensor = F.softmax(pred, dim=1)

    target, target_mask = mask_ignore_pixels(target, ignore_index)

    # create the labels one hot torch.Tensor. A half-precision target is built in float32, the dtype of the sums below:
    # the intersection gradient of a class absent from the target is 2 / (cardinality + eps) up to the averaging, past
    # the float16 range once that class's probabilities underflow, and float16 inf times the zero target is NaN.
    target_dtype = torch.float32 if pred.dtype in (torch.float16, torch.bfloat16) else pred.dtype
    target_one_hot: torch.Tensor = one_hot(target, num_classes=pred.shape[1], device=pred.device, dtype=target_dtype)

    # mask ignore pixels
    if target_mask is not None:
        target_mask.unsqueeze_(1)
        target_one_hot = target_one_hot * target_mask
        pred_soft = pred_soft * target_mask

    # compute the actual dice score
    if weight is not None:
        KORNIA_CHECK_IS_TENSOR(weight, "weight must be torch.Tensor or None.")
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

    output_dtype = torch.promote_types(pred.dtype, weight.dtype)

    # set dimensions for the appropriate averaging
    dims: tuple[int, ...] = (2, 3)

    # The weighted micro Dice is 2 sum(w p t) / sum(w (p + t)): the weight enters the intersection once, through the
    # weighted scores, so the intersection pairs them with the unweighted target.
    intersection_target = target_one_hot
    if average == "micro":
        dims = (1, *dims)

        weight = weight.view(-1, 1, 1)
        pred_soft = pred_soft * weight
        target_one_hot = target_one_hot * weight

    # Half-precision pixel counts can overflow before the Dice ratio is formed.
    reduction_dtype = torch.float32 if pred_soft.dtype in (torch.float16, torch.bfloat16) else pred_soft.dtype
    intersection = torch.sum(pred_soft * intersection_target, dims, dtype=reduction_dtype)
    cardinality = torch.sum(pred_soft + target_one_hot, dims, dtype=reduction_dtype)

    # An empty weighted reduction can occur for fully ignored samples or when all contributing class weights are zero.
    # Make the denominator safe before division so the unused branch cannot introduce NaNs into the backward pass.
    empty = cardinality == 0
    dice_score = (2.0 * intersection / (cardinality + eps).masked_fill(empty, 1.0)).masked_fill(empty, 0.0)
    dice_loss = -dice_score + 1.0

    # reduce the loss across samples (and classes in case of `macro` averaging)
    if average == "macro":
        # A class is present in a sample when one of the sample's non-ignored target pixels has it.
        present = (target_one_hot == 1).any(dim=-1).any(dim=-1)
        weight = weight * present
        normalizer = weight.sum(-1)
        # No present class with a nonzero weight (all pixels ignored, or weight 0): loss 1, as before.
        empty = normalizer == 0
        dice_loss = (dice_loss * weight).sum(-1) / normalizer.masked_fill(empty, 1)
        dice_loss = dice_loss.masked_fill(empty, 1)

    return torch.mean(dice_loss).to(output_dtype)


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    Convention:
        See the Convention block of :func:`~kornia.losses.dice_loss`.

    Args:
        average: Reduction strategy for multi-class computation. Use "micro" to pool the classes of each
            sample, or "macro" to average over classes present in each sample's non-ignored target.
        eps: Small constant added to the denominator for numerical stability.
        weight: Optional class-weight tensor of shape :math:`(C,)`.
        ignore_index: Label value to exclude from loss computation.

    Shapes:
        - pred: :math:`(N, C, H, W)` where C is the number of classes.
        - target: :math:`(N, H, W)` where each value is in the range :math:`[0, C-1]`.
        - Output: scalar.
    """

    def __init__(
        self,
        average: str = "micro",
        eps: float = 1e-8,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = -100,
    ) -> None:
        """Initialize the Dice loss module.

        Args:
            average: Reduction strategy for multi-class computation.
            eps: Small constant added for numerical stability.
            weight: Optional class-weight tensor of shape :math:`(C,)`.
            ignore_index: Label value to exclude from loss computation.
        """
        super().__init__()
        self.average = average
        self.eps = eps
        self.register_buffer("weight", weight, persistent=False)
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Sørensen-Dice loss for segmentation logits and labels.

        Args:
            pred: Logit tensor with shape :math:`(B, C, H, W)`, where
                :math:`B` is batch size, :math:`C` is number of classes,
                :math:`H` is height, and :math:`W` is width.
            target: Integer target labels with shape :math:`(B, H, W)`.

        Returns:
            Scalar tensor containing ``1 - Dice`` after applying the configured
            averaging strategy, class weights, numerical epsilon, and optional
            ignored label handling.
        """
        return dice_loss(pred, target, self.average, self.eps, self.weight, self.ignore_index)
