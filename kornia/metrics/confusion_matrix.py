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

import torch

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR
from kornia.core.utils import is_compiling

# Inspired by:
# https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py#L68-L73


def _check_labels(name: str, labels: torch.Tensor) -> None:
    KORNIA_CHECK_IS_TENSOR(labels, f"Input {name} must be a tensor of integer labels")
    # bool masks count as 0/1 labels, as they did before the check.
    KORNIA_CHECK(
        not (labels.dtype.is_floating_point or labels.dtype.is_complex),
        f"Input {name} must have an integer dtype. Got {labels.dtype}",
    )


def _check_label_range(name: str, labels: torch.Tensor, num_classes: int) -> None:
    low, high = int(labels.min()), int(labels.max())
    KORNIA_CHECK(
        0 <= low and high < num_classes,
        f"Input {name} must contain values in [0, {num_classes}). Got values in [{low}, {high}]",
    )


def confusion_matrix(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, normalized: bool = False
) -> torch.Tensor:
    r"""Compute confusion matrix to evaluate the accuracy of a classification.

    Convention:
        - ``pred`` and ``target`` are class labels of the same shape, prediction first, in ``uint8`` or any signed
          integer dtype; a bool tensor counts as the labels 0 and 1. A floating-point label tensor raises, and so does
          a label outside :math:`[0, K)`, unless graph capture traces the range check (``torch.export``,
          ``torch.compile(fullgraph=True)``), which skips it: an out-of-range label is then counted in another cell
          or makes the call fail.
        - The first axis is always the batch: the result holds one :math:`(K, K)` float32 count matrix per sample,
          :math:`(B, K, K)`, and is never pooled over the batch, so a flat :math:`(N,)` label vector gives :math:`N`
          matrices that count one label each. Sum over the first axis for the matrix of a whole batch or dataset.
        - Rows are the target and columns the prediction, ``cm[b, target, pred]``; swapping the two arguments
          transposes every count matrix. :ref:`Losses and metrics <losses-metrics-conventions>` maps the matrix and
          ``normalized`` onto scikit-learn.

    Args:
        pred: tensor with estimated targets returned by a
          classifier. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1.
        target: tensor with ground truth (correct) target
          values. The shape must be that of ``pred``, and it must contain integer
          values between 0 and K-1.
        num_classes: total possible number of classes in target.
        normalized: whether to normalize each target row by its sum plus ``1e-6``.
          Non-empty rows sum approximately to one; empty rows remain zero.

    Returns:
        a tensor containing the confusion matrix with shape
        :math:`(B, K, K)` where K is the number of classes.

    Example:
        >>> logits = torch.tensor([[0, 1, 0]])
        >>> target = torch.tensor([[0, 1, 0]])
        >>> confusion_matrix(logits, target, num_classes=3)
        tensor([[[2., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 0.]]])

    """
    _check_labels("pred", pred)
    _check_labels("target", target)
    if not pred.shape == target.shape:
        raise ValueError(f"Inputs pred and target must have the same shape. Got: {pred.shape} and {target.shape}")
    if not pred.device == target.device:
        raise ValueError(f"Inputs must be in the same device. Got: {pred.device} - {target.device}")

    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"The number of classes must be an integer of at least two. Got: {num_classes}")

    batch_size: int = pred.shape[0]
    # An empty batch used to fail on view(0, -1) because a zero-numel tensor
    # cannot infer the trailing dimension. Return an empty (0, K, K) matrix.
    if batch_size == 0:
        return torch.zeros(0, num_classes, num_classes, device=pred.device, dtype=torch.float32)

    # The range check reads the data, which graph capture cannot do; skip it under any capture.
    if not is_compiling() and pred.numel() > 0:
        _check_label_range("pred", pred, num_classes)
        _check_label_range("target", target, num_classes)

    # Form indices in int64: uint8 labels would otherwise wrap from num_classes = 17.
    indices = (pred.long() + target.long() * num_classes).reshape(batch_size, -1)
    # A fixed-size histogram avoids per-sample bincount launches and their dynamic output sizes.
    # Accumulate integers before converting to float32 so large counts do not lose increments.
    num_cells = num_classes * num_classes
    # On CUDA every pixel is a global atomic add, and pixels sharing a cell serialize. Spread consecutive
    # pixels over up to 256 copies of the histogram (at most 16384 cells per sample) and sum the copies.
    # The count depends on the class count only, so captured graphs stay free of shape guards.
    copies = max(1, min(256, 16384 // num_cells)) if indices.is_cuda else 1
    if copies > 1:
        indices = indices + (torch.arange(indices.shape[1], device=indices.device) % copies) * num_cells
    confusion_vec = torch.zeros(batch_size, copies * num_cells, device=pred.device, dtype=torch.int64)
    confusion_vec.scatter_add_(1, indices, torch.ones_like(indices))
    if copies > 1:
        confusion_vec = confusion_vec.view(batch_size, copies, num_cells).sum(1)
    confusion_mat: torch.Tensor = confusion_vec.view(batch_size, num_classes, num_classes).to(torch.float32)  # BxKxK

    if normalized:
        norm_val: torch.Tensor = torch.sum(confusion_mat, dim=2, keepdim=True)
        confusion_mat = confusion_mat / (norm_val + 1e-6)

    return confusion_mat
