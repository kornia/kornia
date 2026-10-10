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

    # hack for bitcounting 2 arrays together
    # NOTE: torch.bincount does not implement batched version
    # The cell index is formed in int64: in the labels' own dtype it wraps, for uint8 from num_classes = 17.
    pre_bincount: torch.Tensor = pred.long() + target.long() * num_classes
    pre_bincount_vec: torch.Tensor = pre_bincount.reshape(batch_size, -1)

    confusion_list = []
    for iter_id in range(batch_size):
        pb: torch.Tensor = pre_bincount_vec[iter_id]
        bin_count: torch.Tensor = torch.bincount(pb, minlength=num_classes**2)
        confusion_list.append(bin_count)

    confusion_vec: torch.Tensor = torch.stack(confusion_list)
    confusion_mat: torch.Tensor = confusion_vec.view(batch_size, num_classes, num_classes).to(torch.float32)  # BxKxK

    if normalized:
        norm_val: torch.Tensor = torch.sum(confusion_mat, dim=2, keepdim=True)
        confusion_mat = confusion_mat / (norm_val + 1e-6)

    return confusion_mat
