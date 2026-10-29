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

import sys
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK
from kornia.core.utils import is_autocast_enabled, is_compiling, is_exporting
from kornia.losses._utils import mask_ignore_pixels
from kornia.losses.one_hot import _check_class_labels, one_hot

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

# A module constant: torch.compile on torch 2.5 cannot read sys.float_info inside a traced function.
_FLOAT_MAX: float = sys.float_info.max


def _tversky_cpu_eligible(pred: torch.Tensor) -> bool:
    if torch.jit.is_scripting():
        return False
    return (
        pred.device.type == "cpu"
        and pred.dtype in (torch.float32, torch.float64)
        and not is_compiling()
        and not is_exporting()
        and not torch.jit.is_tracing()
        # Small images keep the cheaper one-hot route; measured forward + backward on CPU.
        and pred.numel() >= 1 << 15
        and not torch._C._are_functorch_transforms_active()
        and not is_autocast_enabled()
    )


def _tversky_loss_cpu(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float,
    ignore_index: Optional[int],
) -> torch.Tensor:
    """Accumulate target-class terms without allocating a one-hot image."""
    batch, classes = pred.shape[:2]
    target, mask = mask_ignore_pixels(target, ignore_index)
    KORNIA_CHECK(target.dtype == torch.int64, f"labels must be of dtype torch.int64. Got: {target.dtype}")
    target = _check_class_labels(target, classes)
    probability = pred.softmax(1)
    target_probability = probability.gather(1, target.unsqueeze(1)).squeeze(1)
    missed_probability = 1.0 - target_probability
    non_target = probability.clone()
    non_target.scatter_(1, target.unsqueeze(1), 0.0)
    valid = torch.ones_like(target, dtype=pred.dtype) if mask is None else mask.to(pred.dtype)
    if mask is not None:
        target_probability = target_probability * valid
        missed_probability = missed_probability * valid
        non_target.mul_(mask.unsqueeze(1))

    index = target.flatten(1)
    # As in the CPU Dice path, double bins avoid long float32 scatter sums losing small contributions.
    bins = pred.new_zeros(batch, classes, dtype=torch.float64)
    tp = bins.scatter_add(1, index, target_probability.flatten(1).to(torch.float64)).to(pred.dtype)
    fn = bins.scatter_add(1, index, missed_probability.flatten(1).to(torch.float64)).to(pred.dtype)
    present = bins.scatter_add(1, index, valid.flatten(1).to(torch.float64)) > 0
    # Sum false positives directly: mass - tp would cancel the small errors of a nearly correct prediction.
    fp = non_target.sum((2, 3))
    denominator = tp + alpha * fp + beta * fn + eps
    score = (tp / denominator.masked_fill(~present, 1.0)).masked_fill(~present, 0.0)
    score = score.sum(-1) / present.sum(-1).clamp_min(1)
    return (1.0 - score.mean()).to(pred.dtype)


def _check_parameters(alpha: float, beta: float, eps: float) -> None:
    # Only Python numbers are checked: a tensor weight (a 0-d device tensor, or one weight per class) worked before,
    # and reading its value here would cost a host sync on every call and break a fullgraph compile. Comparisons, not
    # math.isfinite: torch.compile traces a float that changes between calls symbolically, and math.isfinite on it
    # breaks the graph. NaN fails both comparisons; a bound of `< math.inf` would be dropped from the compiled guards.
    for name, value in (("alpha", alpha), ("beta", beta), ("eps", eps)):
        if isinstance(value, (int, float)) and not (0.0 <= value <= _FLOAT_MAX):
            raise ValueError(f"{name} must be non-negative and finite. Got: {value}")


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
        Large eager CPU float32/float64 inputs avoid a dense one-hot target;
        their class sums agree with the one-hot computation to rounding.

    Example:
        >>> N = 5  # num_classes
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = tversky_loss(pred, target, alpha=0.5, beta=0.5)
        >>> output.backward()

    """
    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"pred type is not a torch.Tensor. Got {type(pred)}")

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"target type is not a torch.Tensor. Got {type(target)}")

    if not len(pred.shape) == 4:
        raise ValueError(f"Invalid pred shape, we expect BxNxHxW. Got: {pred.shape}")

    if not pred.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"pred and target shapes must be the same. Got: {pred.shape} and {target.shape}")

    if not pred.device == target.device:
        raise ValueError(f"pred and target must be in the same device. Got: {pred.device} and {target.device}")

    if not (pred.shape[0] == target.shape[0] and pred.shape[2:] == target.shape[1:]):
        raise ValueError(f"Expected target size {torch.Size((pred.shape[0], *pred.shape[2:]))}, got {target.shape}")

    _check_parameters(alpha, beta, eps)

    if _tversky_cpu_eligible(pred):
        return _tversky_loss_cpu(pred, target, alpha, beta, eps, ignore_index)

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
        _check_parameters(alpha, beta, eps)
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
