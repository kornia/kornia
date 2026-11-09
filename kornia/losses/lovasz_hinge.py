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

import torch
from torch import Tensor, nn

from kornia.core.check import KORNIA_CHECK_SHAPE

# based on:
# https://github.com/bermanmaxim/LovaszSoftmax


def lovasz_hinge_loss(pred: Tensor, target: Tensor) -> Tensor:
    r"""Compute the binary Lovász-hinge IoU loss of ``(B, 1, H, W)`` logits against ``(B, H, W)`` labels.

    ``loss = lovasz_hinge_loss(pred, target)`` optimises the intersection over union (IoU) of two foregrounds: the
    one predicted by the single logit channel of ``pred`` (:math:`X`) and the one labelled in ``target`` (:math:`Y`),
    through the Lovász extension [1] of the Jaccard loss [2].

    .. math::

        \Delta_J = 1 - \frac{|X \cap Y|}{|X \cup Y|}

    Reference:
        [1] http://proceedings.mlr.press/v37/yub15.pdf
        [2] https://arxiv.org/pdf/1705.08790.pdf

    Convention:
        - ``pred`` holds one logit channel and ``target`` labels of 1 for the foreground and 0 for the background; the
          batch and spatial sizes must match. The loss is computed per image and averaged over the batch; there is no
          ``reduction``.
        - The extension is evaluated at the hinge errors :math:`\max(0, 1 - z (2t - 1))` of logit :math:`z` and label
          :math:`t`, so the loss is not :math:`1 - \text{IoU}`: logits of :math:`\pm 1` that are
          positive on the predicted foreground give :math:`2 (1 - \text{IoU})` of that prediction, and correct
          logits of magnitude at least 1 give 0.
        - ``target`` is not validated and there is no ``ignore_index``: a label of 2 is accepted, and ``-100``, the
          ``ignore_index`` of :func:`~kornia.losses.focal_loss` and :func:`~kornia.losses.dice_loss`, can make the
          loss negative.

    Args:
        pred: logits ``(B, 1, H, W)``.
        target: binary labels ``(B, H, W)``.

    Returns:
        a scalar with the computed loss, in the dtype of a floating-point ``pred`` and in float32 for an
        integer or bool ``pred``. The Jaccard weights and the sum over pixels are computed in float32 for a
        float16 or bfloat16 ``pred``.

    See Also:
        - :class:`~kornia.losses.LovaszHingeLoss`: the module form.
        - :func:`~kornia.losses.lovasz_softmax_loss`: the multi-class loss, from ``C >= 2`` logit channels against
          class indices.

    Example:
        >>> N = 1  # num_classes
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = lovasz_hinge_loss(pred, target)
        >>> output.backward()

    """
    KORNIA_CHECK_SHAPE(pred, ["B", "1", "H", "W"])

    KORNIA_CHECK_SHAPE(target, ["B", "H", "W"])

    if not (pred.shape[0] == target.shape[0] and pred.shape[-2:] == target.shape[-2:]):
        raise ValueError(f"pred and target shapes must be the same. Got: {pred.shape} and {target.shape}")

    if not pred.device == target.device:
        raise ValueError(f"pred and target must be in the same device. Got: {pred.device} and {target.device}")

    # flatten pred and target [B, -1] and to float
    # The labels, the Jaccard weights and the sum over pixels are accumulated in the prediction dtype, or in float32
    # for a half-precision prediction, where pixel counts stay exact up to 2**24 pixels.
    accumulation_dtype = torch.promote_types(pred.dtype, torch.float32)
    pred_flatten: Tensor = pred.flatten(start_dim=1)
    target_flatten: Tensor = target.flatten(start_dim=1).to(accumulation_dtype)

    # get shapes
    B, N = pred_flatten.shape

    # compute actual loss
    signs = 2.0 * target_flatten - 1.0
    errors = 1.0 - pred_flatten * signs
    errors_sorted, permutation = errors.sort(dim=1, descending=True)
    batch_index: Tensor = torch.arange(B, device=pred.device).reshape(-1, 1).repeat(1, N).reshape(-1)
    target_sorted: Tensor = target_flatten[batch_index, permutation.view(-1)]
    target_sorted = target_sorted.view(B, N)
    target_sorted_sum: Tensor = target_sorted.sum(1, keepdim=True)
    intersection: Tensor = target_sorted_sum - target_sorted.cumsum(1)
    union: Tensor = target_sorted_sum + (1.0 - target_sorted).cumsum(1)
    gradient: Tensor = 1.0 - intersection / union
    if N > 1:
        gradient[..., 1:] = gradient[..., 1:] - gradient[..., :-1]
    loss: Tensor = (errors_sorted.relu() * gradient).sum(1).mean()
    # an integer or bool pred keeps the float32 loss instead of truncating it
    return loss.to(pred.dtype if pred.is_floating_point() else accumulation_dtype)


class LovaszHingeLoss(nn.Module):
    r"""Compute the binary Lovász-hinge IoU loss of ``(B, 1, H, W)`` logits against ``(B, H, W)`` labels (module form).

    ``criterion = LovaszHingeLoss()`` takes no arguments, and ``loss = criterion(pred, target)`` is
    :func:`~kornia.losses.lovasz_hinge_loss`.

    Convention:
        See the Convention block of :func:`~kornia.losses.lovasz_hinge_loss`.

    Shape:
        - pred: logits ``(B, 1, H, W)``.
        - target: binary labels ``(B, H, W)``.
        - Output: a scalar.

    See Also:
        - :func:`~kornia.losses.lovasz_hinge_loss`: the one-call form.
        - :class:`~kornia.losses.LovaszSoftmaxLoss`: the multi-class loss, from ``C >= 2`` logit channels against
          class indices.

    Example:
        >>> N = 1  # num_classes
        >>> criterion = LovaszHingeLoss()
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(pred, target)
        >>> output.backward()

    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute binary Lovasz hinge loss for segmentation logits.

        Args:
            pred: Binary logit tensor with shape ``(B, 1, H, W)``.
            target: Binary label tensor with shape ``(B, H, W)``.

        Returns:
            Scalar tensor containing the Lovasz hinge surrogate for the
            intersection-over-union objective.
        """
        return lovasz_hinge_loss(pred=pred, target=target)
