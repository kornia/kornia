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
from torch import Tensor, nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

# based on:
# https://github.com/bermanmaxim/LovaszSoftmax


def lovasz_softmax_loss(pred: Tensor, target: Tensor, weight: Optional[Tensor] = None) -> Tensor:
    r"""Compute the multi-class Lovász-softmax IoU loss of ``(B, C, H, W)`` logits against ``(B, H, W)`` class indices.

    ``loss = lovasz_softmax_loss(pred, target)`` optimises the intersection over union (IoU) of every class: the
    softmax of ``pred`` over the classes (:math:`X`) against the one-hot ``target`` (:math:`Y`). It averages over the
    classes the Lovász extension of the Jaccard loss of each class [1], evaluated at the errors :math:`|X - Y|`.

    .. math::

        \Delta_J = 1 - \frac{|X \cap Y|}{|X \cup Y|}

    Reference:
        [1] https://arxiv.org/pdf/1705.08790.pdf

    Convention:
        - ``pred`` holds logits ``(B, C, H, W)`` with ``C >= 2`` and the softmax over dim 1 is taken inside;
          ``target`` holds class labels ``(B, H, W)`` in ``[0, C)`` with the spatial size of ``pred``. The batch
          sizes match, or ``pred`` holds one image, which is scored against every target. There is no
          ``ignore_index`` and the labels are not validated: a label of ``-100`` is not ignored, and the result
          for it is undefined.
        - Each image is scored on its own and the losses are averaged over the batch; there is no ``reduction``. An
          image's loss is the mean over all ``C`` classes: a class absent from the image enters it with that image's
          largest probability for the class.
        - ``weight`` multiplies the term of each class and the mean still divides by ``C``, so a uniform weight of 2
          doubles the loss, where ``dice_loss(average='macro')`` divides by the summed weights of the classes it
          averages.

    Args:
        pred: logits ``(B, C, H, W)`` with ``C >= 2``.
        target: class indices ``(B, H, W)`` in ``[0, C)``.
        weight: class weights ``(C,)``.

    Returns:
        a scalar with the computed loss, in the dtype of ``pred`` promoted with the dtype of ``weight``.
        The Jaccard weights and the sum over pixels are computed in float32 for a float16 or bfloat16 ``pred``.

    See Also:
        - :class:`~kornia.losses.LovaszSoftmaxLoss`: the module form.
        - :func:`~kornia.losses.lovasz_hinge_loss`: the binary loss, from one logit channel.
        - :func:`~kornia.losses.dice_loss`: optimises the Dice overlap and drops ignored pixels.

    Example:
        >>> N = 5  # num_classes
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = lovasz_softmax_loss(pred, target)
        >>> output.backward()

    """
    KORNIA_CHECK_SHAPE(pred, ["B", "N", "H", "W"])

    KORNIA_CHECK_SHAPE(target, ["B", "H", "W"])

    if not pred.shape[1] > 1:
        raise ValueError(f"Invalid pred shape, we expect BxNxHxW, with N > 1. Got: {pred.shape}")

    # A prediction batch of one is scored against every target, as the computation below broadcasts it.
    if not ((pred.shape[0] == target.shape[0] or pred.shape[0] == 1) and pred.shape[-2:] == target.shape[-2:]):
        raise ValueError(f"pred and target shapes must be the same. Got: {pred.shape} and {target.shape}")

    if not pred.device == target.device:
        raise ValueError(f"pred and target must be in the same device. Got: {pred.device} and {target.device}")

    num_of_classes = pred.shape[1]
    # compute the actual dice score
    if weight is not None:
        KORNIA_CHECK_IS_TENSOR(weight, "weight must be Tensor or None.")
        KORNIA_CHECK(
            (weight.shape[0] == num_of_classes and weight.numel() == num_of_classes),
            f"weight must hold one value per class of pred: got weight of shape {weight.shape} for pred of shape "
            f"{pred.shape}",
        )
        KORNIA_CHECK(
            weight.device == pred.device,
            f"weight and pred must be in the same device. Got: {weight.device} and {pred.device}",
        )

    # flatten pred to [B, C, H*W] and target to [B, H*W]; flatten keeps an empty batch, where reshape(B, -1) cannot
    # infer the -1
    pred_flatten: Tensor = pred.flatten(start_dim=2)
    target_flatten: Tensor = target.flatten(start_dim=1)

    # get shapes
    _, C, N = pred_flatten.shape

    # compute softmax over the classes axis
    pred_soft: Tensor = pred_flatten.softmax(1)

    # compute actual loss
    foreground: Tensor = (
        torch.nn.functional.one_hot(target_flatten.to(torch.int64), num_classes=C).permute(0, 2, 1).to(pred.dtype)
    )
    errors: Tensor = (pred_soft - foreground).abs()
    errors_sorted, permutations = torch.sort(errors, dim=2, descending=True)
    # The Jaccard gradient uses the foreground indicator for each class, not the class labels.
    # Pixel counts, the Jaccard weights and the sum over pixels are accumulated in the prediction dtype, or in
    # float32 for a half-precision prediction, where counts stay exact up to 2**24 pixels.
    accumulation_dtype = torch.promote_types(pred.dtype, torch.float32)
    target_sorted = foreground.gather(2, permutations).to(accumulation_dtype)
    target_sorted_sum = target_sorted.sum(2, keepdim=True)
    intersection = target_sorted_sum - target_sorted.cumsum(2)
    union = target_sorted_sum + (1.0 - target_sorted).cumsum(2)
    gradient = 1.0 - intersection / union
    if N > 1:
        gradient[..., 1:] = gradient[..., 1:] - gradient[..., :-1]
    weighted_errors = errors_sorted * gradient
    loss_per_class = weighted_errors.sum(2).mean(0)
    output_dtype = pred.dtype
    if weight is not None:
        loss_per_class = loss_per_class * weight
        output_dtype = torch.promote_types(output_dtype, weight.dtype)
    final_loss: Tensor = loss_per_class.mean()
    return final_loss.to(output_dtype)


class LovaszSoftmaxLoss(nn.Module):
    r"""Compute the multi-class Lovász-softmax IoU loss of ``(B, C, H, W)`` logits against class indices (module form).

    ``criterion = LovaszSoftmaxLoss(weight)`` stores ``weight``, and ``loss = criterion(pred, target)`` is
    :func:`~kornia.losses.lovasz_softmax_loss`. A plain tensor ``weight`` is stored as a non-persistent buffer,
    so ``criterion.to(device)`` moves it and ``state_dict`` omits it.

    Convention:
        See the Convention block of :func:`~kornia.losses.lovasz_softmax_loss`.

    Args:
        weight: class weights ``(C,)``; ``None`` weights every class by 1.

    Shape:
        - pred: logits ``(B, C, H, W)`` with ``C >= 2``.
        - target: class indices ``(B, H, W)`` in ``[0, C)``.
        - Output: a scalar.

    See Also:
        - :func:`~kornia.losses.lovasz_softmax_loss`: the one-call form.
        - :class:`~kornia.losses.LovaszHingeLoss`: the binary loss, from one logit channel.
        - :class:`~kornia.losses.DiceLoss`: optimises the Dice overlap and drops ignored pixels.

    Example:
        >>> N = 5  # num_classes
        >>> criterion = LovaszSoftmaxLoss()
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(pred, target)
        >>> output.backward()

    """

    def __init__(self, weight: Optional[Tensor] = None) -> None:
        super().__init__()
        if isinstance(weight, nn.Parameter):
            self.weight = weight
        else:
            self.register_buffer("weight", weight, persistent=False)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute multi-class Lovasz-Softmax loss.

        Args:
            pred: Class logit tensor with shape ``(B, C, H, W)``, where
                :math:`C` is the number of classes.
            target: Integer class-label tensor with shape ``(B, H, W)``.

        Returns:
            Scalar tensor containing the Lovasz-Softmax surrogate for the
            mean intersection-over-union objective, using ``self.weight`` when
            class weights are configured.
        """
        return lovasz_softmax_loss(pred=pred, target=target, weight=self.weight)
