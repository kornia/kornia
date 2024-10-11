from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from kornia.losses._utils import mask_ignore_pixels
from kornia.utils.one_hot import one_hot

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py


def tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float = 1e-8,
    ignore_index: Optional[int] = -100,
) -> torch.Tensor:
    r"""Criterion that computes Tversky Coefficient loss.

    According to :cite:`salehi2017tversky`, we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \setminus G| + \beta |G \setminus P|}

    Where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Note:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Args:
        pred: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        target: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C-1`.
        alpha: the first coefficient in the denominator.
        beta: the second coefficient in the denominator.
        eps: scalar for numerical stability.
        ignore_index: labels with this value are ignored in the loss computation.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = tversky_loss(pred, target, alpha=0.5, beta=0.5)
        >>> output.backward()
    """
    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"pred type is not a torch.Tensor. Got {type(pred)}")

    if not len(pred.shape) == 4:
        raise ValueError(f"Invalid pred shape, we expect BxNxHxW. Got: {pred.shape}")

    if not pred.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"pred and target shapes must be the same. Got: {pred.shape} and {target.shape}")

    if not pred.device == target.device:
        raise ValueError(f"pred and target must be in the same device. Got: {pred.device} and {target.device}")

    # compute softmax over the classes axis
    pred_soft: torch.Tensor = F.softmax(pred, dim=1)

    target, target_mask = mask_ignore_pixels(target, ignore_index)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=pred.shape[1], device=pred.device, dtype=pred.dtype)

    # mask ignore pixels
    if target_mask is not None:
        target_mask.unsqueeze_(1)
        target_one_hot = target_one_hot * target_mask
        pred_soft = pred_soft * target_mask

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(pred_soft * target_one_hot, dims)
    fps = torch.sum(pred_soft * (-target_one_hot + 1.0), dims)
    fns = torch.sum((-pred_soft + 1.0) * target_one_hot, dims)

    numerator = intersection
    denominator = intersection + alpha * fps + beta * fns
    tversky_loss = numerator / (denominator + eps)

    return torch.mean(-tversky_loss + 1.0)


class TverskyLoss(nn.Module):
    r"""Criterion that computes Tversky Coefficient loss.

    According to :cite:`salehi2017tversky`, we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \setminus G| + \beta |G \setminus P|}

    Where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Note:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Args:
        alpha: the first coefficient in the denominator.
        beta: the second coefficient in the denominator.
        eps: scalar for numerical stability.
        ignore_index: labels with this value are ignored in the loss computation.

    Shape:
        - Pred: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C-1`.

    Examples:
        >>> N = 5  # num_classes
        >>> criterion = TverskyLoss(alpha=0.5, beta=0.5)
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(pred, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, beta: float, eps: float = 1e-8, ignore_index: Optional[int] = -100) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = eps
        self.ignore_index: Optional[int] = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return tversky_loss(pred, target, self.alpha, self.beta, self.eps, self.ignore_index)
