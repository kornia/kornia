from __future__ import annotations

import torch
from torch import nn

from kornia.core import Tensor, tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.utils.one_hot import one_hot

# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py


def focal_loss(
    input: Tensor,
    target: Tensor,
    alpha: float | None,
    gamma: float = 2.0,
    reduction: str = 'none',
    weight: Tensor | None = None,
) -> Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is an integer
          representing correct classification :math:`target[i] \in [0, C)`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.

    Return:
        the computed loss.

    Example:
        >>> C = 5  # num_classes
        >>> input = torch.randn(1, C, 3, 5, requires_grad=True)
        >>> target = torch.randint(C, (1, 3, 5))
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> output = focal_loss(input, target, **kwargs)
        >>> output.backward()
    """

    KORNIA_CHECK_SHAPE(input, ["B", "C", "*"])
    out_size = (input.shape[0],) + input.shape[2:]
    KORNIA_CHECK(
        (input.shape[0] == target.shape[0] and target.shape[1:] == input.shape[2:]),
        f'Expected target size {out_size}, got {target.shape}',
    )
    KORNIA_CHECK(
        input.device == target.device,
        f"input and target must be in the same device. Got: {input.device} and {target.device}",
    )

    # create the labels one hot tensor
    target_one_hot: Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute softmax over the classes axis
    log_input_soft: Tensor = input.log_softmax(1)

    # compute the actual focal loss
    loss_tmp: Tensor = -torch.pow(1.0 - log_input_soft.exp(), gamma) * log_input_soft * target_one_hot

    num_of_classes = input.shape[1]
    boradcast_dims = [-1] + [1] * len(input.shape[2:])
    if alpha is not None:
        alpha_fac = tensor([1 - alpha] + [alpha] * (num_of_classes - 1), dtype=loss_tmp.dtype, device=loss_tmp.device)
        alpha_fac = alpha_fac.view(boradcast_dims)
        loss_tmp = alpha_fac * loss_tmp

    if weight is not None:
        KORNIA_CHECK_IS_TENSOR(weight, "weight must be Tensor or None.")
        KORNIA_CHECK(
            (weight.shape[0] == num_of_classes and weight.numel() == num_of_classes),
            f'weight shape must be (num_of_classes,): ({num_of_classes},), got {weight.shape}',
        )
        KORNIA_CHECK(
            weight.device == input.device,
            f"weight and input must be in the same device. Got: {weight.device} and {input.device}",
        )

        weight = weight.view(boradcast_dims)
        loss_tmp = weight * loss_tmp

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
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
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is an integer
          representing correct classification :math:`target[i] \in [0, C)`.

    Example:
        >>> C = 5  # num_classes
        >>> input = torch.randn(1, C, 3, 5, requires_grad=True)
        >>> target = torch.randint(C, (1, 3, 5))
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(
        self, alpha: float | None, gamma: float = 2.0, reduction: str = 'none', weight: Tensor | None = None
    ) -> None:
        super().__init__()
        self.alpha: float | None = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.weight: Tensor | None = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.weight)


def binary_focal_loss_with_logits(
    input: Tensor,
    target: Tensor,
    alpha: float | None = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
    pos_weight: Tensor | None = None,
    weight: Tensor | None = None,
) -> Tensor:
    r"""Criterion that computes Binary Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with the same shape as input :math:`(N, C, *)`
          where each value is between 0 and 1.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        pos_weight: a weight of positive examples with shape :math:`(num\_of\_classes,)`.
          It is possible to trade off recall and precision by adding weights to positive examples.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.

    Returns:
        the computed loss.

    Examples:
        >>> C = 3  # num_classes
        >>> input = torch.randn(1, C, 5, requires_grad=True)
        >>> target = torch.randint(2, (1, C, 5))
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> output = binary_focal_loss_with_logits(input, target, **kwargs)
        >>> output.backward()
    """

    KORNIA_CHECK_SHAPE(input, ["B", "C", "*"])
    KORNIA_CHECK(input.shape == target.shape, f'Expected target size {input.shape}, got {target.shape}')
    KORNIA_CHECK(
        input.device == target.device,
        f"input and target must be in the same device. Got: {input.device} and {target.device}",
    )

    log_probs_pos: Tensor = nn.functional.logsigmoid(input)
    log_probs_neg: Tensor = nn.functional.logsigmoid(-input)

    pos_term: Tensor = -log_probs_neg.exp().pow(gamma) * target * log_probs_pos
    neg_term: Tensor = -log_probs_pos.exp().pow(gamma) * (1.0 - target) * log_probs_neg
    if alpha is not None:
        pos_term = alpha * pos_term
        neg_term = (1.0 - alpha) * neg_term

    num_of_classes = input.shape[1]
    boradcast_dims = [-1] + [1] * len(input.shape[2:])
    if pos_weight is not None:
        KORNIA_CHECK_IS_TENSOR(pos_weight, "pos_weight must be Tensor or None.")
        KORNIA_CHECK(
            (pos_weight.shape[0] == num_of_classes and pos_weight.numel() == num_of_classes),
            f'pos_weight shape must be (num_of_classes,): ({num_of_classes},), got {pos_weight.shape}',
        )
        KORNIA_CHECK(
            pos_weight.device == input.device,
            f"pos_weight and input must be in the same device. Got: {pos_weight.device} and {input.device}",
        )

        pos_weight = pos_weight.view(boradcast_dims)
        pos_term = pos_weight * pos_term

    loss_tmp: Tensor = pos_term + neg_term
    if weight is not None:
        KORNIA_CHECK_IS_TENSOR(weight, "weight must be Tensor or None.")
        KORNIA_CHECK(
            (weight.shape[0] == num_of_classes and weight.numel() == num_of_classes),
            f'weight shape must be (num_of_classes,): ({num_of_classes},), got {weight.shape}',
        )
        KORNIA_CHECK(
            weight.device == input.device,
            f"weight and input must be in the same device. Got: {weight.device} and {input.device}",
        )

        weight = weight.view(boradcast_dims)
        loss_tmp = weight * loss_tmp

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class BinaryFocalLossWithLogits(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        pos_weight: a weight of positive examples with shape :math:`(num\_of\_classes,)`.
          It is possible to trade off recall and precision by adding weights to positive examples.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: the same shape as Input :math:`(N, C, *)`
          where each value is between 0 and 1.

    Examples:
        >>> C = 3  # num_classes
        >>> input = torch.randn(1, C, 5, requires_grad=True)
        >>> target = torch.randint(2, (1, C, 5))
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = BinaryFocalLossWithLogits(**kwargs)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(
        self,
        alpha: float | None,
        gamma: float = 2.0,
        reduction: str = 'none',
        pos_weight: Tensor | None = None,
        weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.alpha: float | None = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.pos_weight: Tensor | None = pos_weight
        self.weight: Tensor | None = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return binary_focal_loss_with_logits(
            input, target, self.alpha, self.gamma, self.reduction, self.pos_weight, self.weight
        )
