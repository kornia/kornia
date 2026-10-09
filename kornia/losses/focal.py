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

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.losses._utils import mask_ignore_pixels
from kornia.losses.one_hot import one_hot

# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: Optional[float],
    gamma: float = 2.0,
    reduction: str = "none",
    weight: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = -100,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the softmax probability of the target class.
       - :math:`\alpha_t` is :math:`1 - \alpha` for class 0 and :math:`\alpha` for every other class.

    Convention:
        - ``pred`` holds logits ``(B, C, *)`` and the softmax over dim 1 is taken inside. ``target`` holds int64
          class indices ``(B, *)`` in ``[0, C)``; its batch and spatial sizes must equal those of ``pred``.
        - Class 0 is the background: ``alpha`` weights it by :math:`1 - \alpha` and classes :math:`1, \dots, C - 1`
          by :math:`\alpha`, so a relabelling that moves class 0 changes the loss. ``alpha=None`` weights every class
          by 1. With ``C = 2`` the target slice equals :func:`~kornia.losses.binary_focal_loss_with_logits` of the
          logit difference ``pred[:, 1:] - pred[:, :1]`` and the target ``target[:, None]``, with the same ``alpha``
          and ``gamma``: class 1 is the positive class.
        - The default ``reduction='none'`` returns a ``(B, C, *)`` map whose target slice holds the focal term above
          and whose other slices are 0, also where their log-probability overflows to ``-inf``. ``'mean'`` divides
          its sum by every element of that map, ``C`` times the pixel count, so with ``gamma=0`` and ``alpha=None``
          it is the mean cross entropy divided by ``C``.
        - ``weight`` multiplies the slice of class ``c`` by ``weight[c]``; ``'mean'`` is not normalised by the
          weights.
        - A pixel labelled ``ignore_index`` (default ``-100``) is 0 in every slice and still counts in the ``'mean'``
          denominator, where :func:`~kornia.losses.dice_loss` and :func:`~kornia.losses.tversky_loss` drop it.
          :ref:`Losses and metrics <losses-metrics-conventions>` ports
          :func:`~torch.nn.functional.cross_entropy` to this loss.
        - ``alpha`` outside ``[0, 1]`` and a negative ``gamma`` are not validated.

    Args:
        pred: logits torch.Tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels torch.Tensor with shape :math:`(N, *)` where each value is an integer
          representing correct classification :math:`target[i] \in [0, C)`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]` of classes :math:`1, \dots, C - 1`; class 0 is weighted by
          :math:`1 - \alpha`. ``None`` weights every class by 1.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.
        ignore_index: labels with this value contribute 0 to the loss but stay in the ``'mean'`` denominator.

    Return:
        the computed loss.

    Example:
        >>> C = 5  # num_classes
        >>> pred = torch.randn(1, C, 3, 5, requires_grad=True)
        >>> target = torch.randint(C, (1, 3, 5))
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> output = focal_loss(pred, target, **kwargs)
        >>> output.backward()

    """
    KORNIA_CHECK_SHAPE(pred, ["B", "C", "*"])
    out_size = (pred.shape[0],) + pred.shape[2:]
    KORNIA_CHECK(
        (pred.shape[0] == target.shape[0] and target.shape[1:] == pred.shape[2:]),
        f"Expected target size {out_size}, got {target.shape}",
    )
    KORNIA_CHECK(
        pred.device == target.device,
        f"pred and target must be in the same device. Got: {pred.device} and {target.device}",
    )

    target, target_mask = mask_ignore_pixels(target, ignore_index)

    # create the labels one hot torch.Tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=pred.shape[1], device=pred.device, dtype=pred.dtype)

    # mask ignore pixels
    if target_mask is not None:
        target_mask.unsqueeze_(1)
        target_one_hot = target_one_hot * target_mask
        # Exclude ignored logits before log-softmax: an overflowing log probability times zero is NaN.
        pred = pred.masked_fill(~target_mask, 0.0)

    # compute F.softmax over the classes axis
    log_pred_soft: torch.Tensor = pred.log_softmax(1)

    # compute the actual focal loss
    # For 0 < gamma < 1, x ** gamma has an infinite derivative at x = 0, so a probability that rounds to 1 gives a
    # NaN gradient. The loss term (1 - p) ** gamma * log(p) has a zero derivative at p = 1, so saturated entries take
    # the factor's value from a detached copy and pass no gradient through it.
    base = 1.0 - log_pred_soft.exp()
    saturated = base == 0
    focal_weight = torch.where(saturated, base.detach().pow(gamma), base.masked_fill(saturated, 1.0).pow(gamma))
    # Mask before multiplying: a non-target log probability of -inf otherwise gives NaN values and gradients.
    log_pred_soft = log_pred_soft.masked_fill(target_one_hot == 0, 0.0)
    loss_tmp: torch.Tensor = -focal_weight * log_pred_soft * target_one_hot

    num_of_classes = pred.shape[1]
    broadcast_dims = [-1] + [1] * len(pred.shape[2:])
    if alpha is not None:
        alpha_fac = torch.tensor(
            [1 - alpha] + [alpha] * (num_of_classes - 1), dtype=loss_tmp.dtype, device=loss_tmp.device
        )
        alpha_fac = alpha_fac.view(broadcast_dims)
        loss_tmp = alpha_fac * loss_tmp

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

        weight = weight.view(broadcast_dims)
        loss_tmp = weight * loss_tmp

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the softmax probability of the target class.
       - :math:`\alpha_t` is :math:`1 - \alpha` for class 0 and :math:`\alpha` for every other class.

    Convention:
        See the Convention block of :func:`~kornia.losses.focal_loss`.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]` of classes :math:`1, \dots, C - 1`; class 0 is weighted by
          :math:`1 - \alpha`. ``None`` weights every class by 1.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.
        ignore_index: labels with this value contribute 0 to the loss but stay in the ``'mean'`` denominator.

    Shape:
        - Pred: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is an integer
          representing correct classification :math:`target[i] \in [0, C)`.

    Example:
        >>> C = 5  # num_classes
        >>> pred = torch.randn(1, C, 3, 5, requires_grad=True)
        >>> target = torch.randint(C, (1, 3, 5))
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> output = criterion(pred, target)
        >>> output.backward()

    """

    def __init__(
        self,
        alpha: Optional[float],
        gamma: float = 2.0,
        reduction: str = "none",
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = -100,
    ) -> None:
        super().__init__()
        self.alpha: Optional[float] = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.weight: Optional[torch.Tensor] = weight
        self.ignore_index: Optional[int] = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-class focal loss from class logits.

        Args:
            pred: Logit tensor with shape :math:`(B, C, *)`, where :math:`B`
                is batch size, :math:`C` is number of classes, and ``*`` is
                any number of spatial dimensions.
            target: Integer class-label tensor with shape :math:`(B, *)`.

        Returns:
            Focal-loss tensor reduced according to ``self.reduction``. See :func:`~kornia.losses.focal_loss` for
            the class weighting by ``self.alpha`` and ``self.weight``.
        """
        return focal_loss(pred, target, self.alpha, self.gamma, self.reduction, self.weight, self.ignore_index)


def binary_focal_loss_with_logits(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: Optional[float] = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
    pos_weight: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = -100,
) -> torch.Tensor:
    r"""Criterion that computes Binary Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where, for a target of 0 or 1:
       - :math:`p_t` is the sigmoid probability of the target value.
       - :math:`\alpha_t` is :math:`\alpha` for a target of 1 and :math:`1 - \alpha` for a target of 0.

    Convention:
        - Every element of ``pred`` is an independent binary logit: ``pred`` is ``(B, C, *)`` with at least two
          dimensions and ``target`` has its shape, with values in ``[0, 1]``. The default ``reduction='none'``
          returns that shape; ``'mean'`` and ``'sum'`` reduce it as in :func:`~kornia.losses.focal_loss`.
        - ``alpha`` (default ``0.25``) weights the positive term and :math:`1 - \alpha` the negative term. A
          fractional target :math:`t` weights the two terms by :math:`t` and :math:`1 - t`:
          :math:`t \alpha (1 - p)^\gamma (-\log p) + (1 - t) (1 - \alpha) p^\gamma (-\log (1 - p))`, where :math:`p`
          is the sigmoid of the logit.
        - ``pos_weight`` and ``weight`` are ``(C,)`` vectors applied along dim 1: ``pos_weight[c]`` scales the
          positive term of channel ``c`` and ``weight[c]`` its whole loss.
        - A target entry equal to ``ignore_index`` (default ``-100``) contributes 0 and still counts in the
          ``'mean'`` denominator, as in :func:`~kornia.losses.focal_loss`.
        - :ref:`Losses and metrics <losses-metrics-conventions>` maps this loss onto torchvision's
          ``sigmoid_focal_loss`` and torch's ``pos_weight``.

    Args:
        pred: logits torch.Tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels torch.Tensor with the same shape as pred :math:`(N, C, *)`
          where each value is between 0 and 1.
        alpha: Weighting factor :math:`\alpha \in [0, 1]` of the positive term; the negative term is weighted by
          :math:`1 - \alpha`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        pos_weight: a weight of the positive term of each channel, with shape :math:`(num\_of\_classes,)`.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.
        ignore_index: target entries with this value contribute 0 to the loss but stay in the ``'mean'``
          denominator.

    Returns:
        the computed loss.

    Examples:
        >>> C = 3  # num_classes
        >>> pred = torch.randn(1, C, 5, requires_grad=True)
        >>> target = torch.randint(2, (1, C, 5))
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> output = binary_focal_loss_with_logits(pred, target, **kwargs)
        >>> output.backward()

    """
    KORNIA_CHECK_SHAPE(pred, ["B", "C", "*"])
    KORNIA_CHECK(pred.shape == target.shape, f"Expected target size {pred.shape}, got {target.shape}")
    KORNIA_CHECK(
        pred.device == target.device,
        f"pred and target must be in the same device. Got: {pred.device} and {target.device}",
    )

    log_probs_pos: torch.Tensor = nn.functional.logsigmoid(pred)
    log_probs_neg: torch.Tensor = nn.functional.logsigmoid(-pred)

    target, target_mask = mask_ignore_pixels(target, ignore_index)

    if target_mask is not None:
        #  mask ignore pixels
        log_probs_neg = log_probs_neg * target_mask
        log_probs_pos = log_probs_pos * target_mask

    pos_term: torch.Tensor = -(gamma * log_probs_neg).exp() * target * log_probs_pos
    neg_term: torch.Tensor = -(gamma * log_probs_pos).exp() * (1.0 - target) * log_probs_neg
    if alpha is not None:
        pos_term = alpha * pos_term
        neg_term = (1.0 - alpha) * neg_term

    num_of_classes = pred.shape[1]
    broadcast_dims = [-1] + [1] * len(pred.shape[2:])
    if pos_weight is not None:
        KORNIA_CHECK_IS_TENSOR(pos_weight, "pos_weight must be torch.Tensor or None.")
        KORNIA_CHECK(
            (pos_weight.shape[0] == num_of_classes and pos_weight.numel() == num_of_classes),
            f"pos_weight shape must be (num_of_classes,): ({num_of_classes},), got {pos_weight.shape}",
        )
        KORNIA_CHECK(
            pos_weight.device == pred.device,
            f"pos_weight and pred must be in the same device. Got: {pos_weight.device} and {pred.device}",
        )

        pos_weight = pos_weight.view(broadcast_dims)
        pos_term = pos_weight * pos_term

    loss_tmp: torch.Tensor = pos_term + neg_term
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

        weight = weight.view(broadcast_dims)
        loss_tmp = weight * loss_tmp

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class BinaryFocalLossWithLogits(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where, for a target of 0 or 1:
       - :math:`p_t` is the sigmoid probability of the target value.
       - :math:`\alpha_t` is :math:`\alpha` for a target of 1 and :math:`1 - \alpha` for a target of 0.

    Convention:
        See the Convention block of :func:`~kornia.losses.binary_focal_loss_with_logits`. Unlike the function,
        ``alpha`` has no default here.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]` of the positive term; the negative term is weighted by
          :math:`1 - \alpha`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        pos_weight: a weight of the positive term of each channel, with shape :math:`(num\_of\_classes,)`.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.
        ignore_index: target entries with this value contribute 0 to the loss but stay in the ``'mean'``
          denominator.

    Shape:
        - Pred: :math:`(N, C, *)` where C = number of classes.
        - Target: the same shape as Pred :math:`(N, C, *)`
          where each value is between 0 and 1.

    Examples:
        >>> C = 3  # num_classes
        >>> pred = torch.randn(1, C, 5, requires_grad=True)
        >>> target = torch.randint(2, (1, C, 5))
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = BinaryFocalLossWithLogits(**kwargs)
        >>> output = criterion(pred, target)
        >>> output.backward()

    """

    def __init__(
        self,
        alpha: Optional[float],
        gamma: float = 2.0,
        reduction: str = "none",
        pos_weight: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = -100,
    ) -> None:
        super().__init__()
        self.alpha: Optional[float] = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.pos_weight: Optional[torch.Tensor] = pos_weight
        self.weight: Optional[torch.Tensor] = weight
        self.ignore_index: Optional[int] = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute binary focal loss from logits.

        Args:
            pred: Logit tensor with shape :math:`(B, C, *)`, at least two dimensions.
            target: Binary target tensor with the same shape as ``pred``.

        Returns:
            Binary focal-loss tensor reduced according to ``self.reduction``.
            Positive terms may be reweighted by ``self.pos_weight``, and entries
            matching ``self.ignore_index`` contribute 0 when configured.
        """
        return binary_focal_loss_with_logits(
            pred, target, self.alpha, self.gamma, self.reduction, self.pos_weight, self.weight, self.ignore_index
        )
