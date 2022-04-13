from typing import List

import torch
import torch.nn as nn

# based on:
# https://github.com/bermanmaxim/LovaszSoftmax


def lovasz_softmax_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        input: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes > 1.
        labels: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`.

    Return:
        a scalar with the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = lovasz_softmax_loss(input, target)
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Target type is not a torch.Tensor. Got {type(target)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {input.shape}")

    if not len(target.shape) == 3:
        raise ValueError(f"Invalid target shape, we expect BxHxW. Got: {target.shape}")

    if not input.shape[1] > 1:
        raise ValueError(f"Invalid input shape, we expect BxNxHxW, with N > 1. Got: {input.shape}")

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"input and target shapes must be the same. Got: {input.shape} and {target.shape}")

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # flatten input [B, C, -1] and target [B, -1] and to float
    input_flatten: torch.Tensor = input.reshape(input.shape[0], input.shape[1], -1)
    target_flatten: torch.Tensor = target.reshape(target.shape[0], -1).float()

    # get shapes
    B, C, N = input_flatten.shape

    # compute softmax over the classes axis
    input_soft: torch.Tensor = input_flatten.softmax(1)

    # compute actual loss
    losses: List[torch.Tensor] = []
    batch_index: torch.Tensor = torch.arange(B, device=input.device).repeat_interleave(N, dim=0)
    for c in range(C):
        foreground: torch.Tensor = (target_flatten == c)
        class_pred: torch.Tensor = input_soft[:, c]
        errors = (foreground - class_pred).abs()
        errors_sorted, permutation = torch.sort(errors, dim=1, descending=True)
        target_sorted: torch.Tensor = target_flatten[batch_index, permutation.view(-1)]
        target_sorted: torch.Tensor = target_sorted.view(B, N)
        target_sorted_sum: torch.Tensor = target_sorted.sum(dim=1, keepdim=True)
        intersection: torch.Tensor = target_sorted_sum - target_sorted.cumsum(dim=1)
        union: torch.Tensor = target_sorted_sum + (1. - target_sorted).cumsum(dim=1)
        gradient: torch.Tensor = 1. - intersection / union
        if N > 1:
            gradient[..., 1:] = gradient[..., 1:] - gradient[..., :-1]
        loss: torch.Tensor = (errors_sorted.relu() * gradient).sum(dim=1).mean()
        losses.append(loss)
    final_loss: torch.Tensor = torch.stack(losses, dim=0).mean()
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
        input: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes > 1.
        labels: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`.

    Return:
        a scalar with the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> criterion = LovaszSoftmaxLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return lovasz_softmax_loss(input=input, target=target)
