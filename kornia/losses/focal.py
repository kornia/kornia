from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.utils.one_hot import one_hot

# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py


def focal_loss(
    input: Tensor,
    target: Tensor,
    alpha: float | Tensor | None,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: float | None = None,
) -> Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: If `alpha` is `float`, used as weighting factor :math:`\alpha \in [0, 1]`.
          If `alpha` is `torch.Tensor`, used as the weights for classes,
          and the size of `alpha` should be (num_of_classes,).
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    KORNIA_CHECK_SHAPE(input, ["B", "C", "*"])

    n = input.shape[0]
    out_size = (n,) + input.shape[2:]

    KORNIA_CHECK(target.shape[1:] == input.shape[2:], f'Expected target size {out_size}, got {target.size()}')
    KORNIA_CHECK(
        input.device == target.device,
        f"input and target must be in the same device. Got: {input.device} and {target.device}",
    )

    # create the labels one hot tensor
    target_one_hot: Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute softmax over the classes axis
    log_input_soft: Tensor = input.log_softmax(1)

    # compute the actual focal loss
    loss_tmp = -torch.pow(1.0 - log_input_soft.exp(), gamma) * log_input_soft * target_one_hot
    if alpha is not None:
        num_of_classes = input.shape[1]
        if isinstance(alpha, float):
            alpha_fac = torch.tensor([1 - alpha] + [alpha] * (num_of_classes - 1))
        elif isinstance(alpha, Tensor) and alpha.shape != (num_of_classes,):
            raise ValueError(
                f"`alpha` shape must be (num_of_classes,), that is, {(num_of_classes,)}. Got {alpha.shape}"
            )
        else:
            raise TypeError(f'Expected the `alpha` be a float | Tensor | None. Got {type(alpha)}.')

        boradcast_dims = [-1] + [1] * len(input.shape[2:])
        alpha_fac = alpha_fac.view(boradcast_dims).to(loss_tmp)
        loss_tmp = loss_tmp * alpha_fac

    loss_tmp = loss_tmp.sum(1)
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
        alpha: If `alpha` is `float`, used as weighting factor :math:`\alpha \in [0, 1]`.
          If `alpha` is `torch.Tensor`, used as the weights for classes,
          and the size of `alpha` should be (num_of_classes,).
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
          used.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(
        self, alpha: float | None, gamma: float = 2.0, reduction: str = 'none', eps: float | None = None
    ) -> None:
        super().__init__()
        self.alpha: float | None = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float | None = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)


def binary_focal_loss_with_logits(
    input: Tensor,
    target: Tensor,
    alpha: float | None = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: float | None = None,
    pos_weight: Tensor | None = None,
) -> Tensor:
    r"""Function that computes Binary Focal loss.

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: input data tensor of arbitrary shape.
        target: the target tensor with shape matching input.
        alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar for numerically stability when dividing. This is no longer used.
        pos_weight: a weight of positive examples.
          It’s possible to trade off recall and precision by adding weights to positive examples.
          Must be a vector with length equal to the number of classes.

    Returns:
        the computed loss.

    Examples:
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> logits = torch.tensor([[[6.325]],[[5.26]],[[87.49]]])
        >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
        >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
        tensor(21.8725)
    """

    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`binary_focal_loss_with_logits` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    KORNIA_CHECK_SHAPE(input, ["B", "C", "*"])
    KORNIA_CHECK(
        input.shape[0] == target.shape[0],
        f'Expected input batch_size ({input.shape[0]}) to match target batch_size ({target.shape[0]}).',
    )

    if pos_weight is None:
        pos_weight = torch.ones([input.shape[1], *([1] * len(input.shape[2:]))], device=input.device, dtype=input.dtype)

    KORNIA_CHECK_IS_TENSOR(pos_weight)
    KORNIA_CHECK(input.shape[1] == pos_weight.shape[0], "Expected pos_weight equals number of classes.")

    log_probs_pos = nn.functional.logsigmoid(input)
    log_probs_neg = nn.functional.logsigmoid(-input)

    # the alpha term is not extracted to save operations
    if alpha is None:
        loss_tmp = (
            -pos_weight * log_probs_neg.exp().pow(gamma) * target * log_probs_pos
            - log_probs_pos.exp().pow(gamma) * (1.0 - target) * log_probs_neg
        )
    else:
        loss_tmp = (
            -alpha * pos_weight * log_probs_neg.exp().pow(gamma) * target * log_probs_pos
            - (1.0 - alpha) * log_probs_pos.exp().pow(gamma) * (1.0 - target) * log_probs_neg
        )

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
        alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        pos_weight: a weight of positive examples.
          It’s possible to trade off recall and precision by adding weights to positive examples.
          Must be a vector with length equal to the number of classes.

    Shape:
        - Input: :math:`(N, *)`.
        - Target: :math:`(N, *)`.

    Examples:
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = BinaryFocalLossWithLogits(**kwargs)
        >>> input = torch.randn(1, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(
        self, alpha: float | None, gamma: float = 2.0, reduction: str = 'none', pos_weight: Tensor | None = None
    ) -> None:
        super().__init__()
        self.alpha: float | None = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.pos_weight: Tensor | None = pos_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return binary_focal_loss_with_logits(
            input, target, self.alpha, self.gamma, self.reduction, pos_weight=self.pos_weight
        )
