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
import torch.nn.functional as F
from torch import nn

from kornia.losses._utils import mask_ignore_pixels
from kornia.losses.one_hot import one_hot

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
       - :math:`P` and :math:`G` are the softmax probabilities and exact one-hot
         targets for each class, restricted to non-ignored pixels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Note:
       - :math:`\alpha = \beta = 0.5` gives :func:`~kornia.losses.dice_loss` with ``average='macro'`` and twice
         the ``eps``.
       - :math:`\alpha = \beta = 1` corresponds to the per-class Tanimoto coefficient.
       - For :math:`\alpha + \beta = 1` and :math:`\alpha > 0`, the unsmoothed
         per-class score is :math:`F_{\sqrt{\beta / \alpha}}`.

    Convention:
        ``pred``, ``target`` and ``ignore_index`` behave as in :func:`~kornia.losses.dice_loss`: logits
        ``(B, C, H, W)`` with the softmax taken inside, int64 class indices ``(B, H, W)`` and ignored pixels removed
        from their own sample. The index is computed per sample and class, averaged over the classes present in the
        sample's non-ignored target, then over the batch into a 0-d tensor; a sample whose pixels are all ignored
        enters the batch mean as loss 1. There is no ``weight``, ``average`` or ``reduction``.

    Args:
        pred: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        target: labels tensor with shape :math:`(N, H, W)` where each value
          is in range :math:`0 ≤ targets[i] ≤ C-1`.
        alpha: the first coefficient in the denominator.
        beta: the second coefficient in the denominator.
        eps: scalar for numerical stability.
        ignore_index: labels with this value are ignored in the loss computation.

    Return:
        the computed loss.

    Note:
        Softmax, spatial reductions and the ratio use float32 for float16 and
        bfloat16 inputs to avoid overflow in large-image sums and ratio gradients.
        The returned loss retains the input dtype.

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

    if not (pred.shape[0] == target.shape[0] and pred.shape[2:] == target.shape[1:]):
        raise ValueError(f"Expected target size {torch.Size((pred.shape[0], *pred.shape[2:]))}, got {target.shape}")

    # Keep the ratio's backward pass in float32 through softmax for half inputs.
    reduction_dtype = torch.float32 if pred.dtype in (torch.float16, torch.bfloat16) else pred.dtype
    pred_soft = F.softmax(pred, dim=1, dtype=reduction_dtype)
    target, target_mask = mask_ignore_pixels(target, ignore_index)

    target_one_hot = one_hot(target, pred.shape[1], device=pred.device, dtype=reduction_dtype, eps=0.0)

    if target_mask is not None:
        mask = target_mask.unsqueeze(1)
        pred_soft = pred_soft * mask
        target_one_hot = target_one_hot * mask

    dims = (2, 3)
    tp = (pred_soft * target_one_hot).sum(dims)
    fp = (pred_soft * (1.0 - target_one_hot)).sum(dims)
    fn = ((1.0 - pred_soft) * target_one_hot).sum(dims)
    present = target_one_hot.sum(dims) > 0
    denominator = tp + alpha * fp + beta * fn + eps
    # Absent classes do not enter the mean; avoid undefined ratios even when eps is zero.
    score = (tp / denominator.masked_fill(~present, 1.0)).masked_fill(~present, 0.0)
    score = score.sum(-1) / present.sum(-1).clamp_min(1)

    return (1.0 - score.mean()).to(pred.dtype)


class TverskyLoss(nn.Module):
    r"""Criterion that computes Tversky Coefficient loss.

    According to :cite:`salehi2017tversky`, we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \setminus G| + \beta |G \setminus P|}

    Where:
       - :math:`P` and :math:`G` are the softmax probabilities and exact one-hot
         targets for each class, restricted to non-ignored pixels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Note:
       - :math:`\alpha = \beta = 0.5` gives :func:`~kornia.losses.dice_loss` with ``average='macro'`` and twice
         the ``eps``.
       - :math:`\alpha = \beta = 1` corresponds to the per-class Tanimoto coefficient.
       - For :math:`\alpha + \beta = 1` and :math:`\alpha > 0`, the unsmoothed
         per-class score is :math:`F_{\sqrt{\beta / \alpha}}`.

    Convention:
        See the Convention block of :func:`~kornia.losses.tversky_loss`.

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
        """Compute the Tversky segmentation loss for class logits and labels.

        Args:
            pred: Logit tensor with shape :math:`(B, C, H, W)`, where
                :math:`B` is batch size, :math:`C` is number of classes,
                :math:`H` is height, and :math:`W` is width.
            target: Integer class-label tensor with shape :math:`(B, H, W)`.
                Labels matching ``self.ignore_index`` are excluded from the
                loss when an ignore index is configured.

        Returns:
            Scalar tensor containing the Tversky loss. ``self.alpha`` controls
            the false-positive penalty and ``self.beta`` controls the
            false-negative penalty.
        """
        return tversky_loss(pred, target, self.alpha, self.beta, self.eps, self.ignore_index)
