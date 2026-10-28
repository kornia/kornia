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

import operator
from typing import Any, Tuple

import torch

from kornia.core.check import (
    KORNIA_CHECK,
    KORNIA_CHECK_IS_TENSOR,
    KORNIA_CHECK_SAME_DEVICE,
    KORNIA_CHECK_SHAPE,
    KORNIA_CHECK_SIZES,
)


def _is_positive_index(k: Any) -> bool:
    """Return whether ``k`` is a positive integer; NumPy integers and 0-d integer tensors count, bools do not."""
    if isinstance(k, bool) or (isinstance(k, torch.Tensor) and k.dtype == torch.bool):
        return False
    try:
        return operator.index(k) > 0
    except (TypeError, ValueError):
        return False


def accuracy(pred: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    """Compute the accuracy over the k top predictions for the specified values of k.

    Convention:
        - ``pred`` holds one score per class, :math:`(B, C)`; only the order of the scores in a row matters.
          ``target`` holds class indices, :math:`(B,)` or :math:`(B, 1)`. The batch sizes of ``pred`` and
          ``target`` must match.
        - The result is a list with one 0-d float32 tensor per entry of ``topk``, in that order. Each is a
          **percentage** in :math:`[0, 100]`, not a fraction, taken over the whole batch: a sample counts for ``k``
          when its target is among its ``k`` highest scores, and a ``k`` above :math:`C` counts as :math:`C`.
          :ref:`Losses and metrics <losses-metrics-conventions>` lists the scale of every task metric.

    Args:
        pred: the class scores, with shape :math:`(B, C)`.
        target: the ground-truth class indices, with shape :math:`(B,)` or :math:`(B, 1)`.
        topk: the expected topk ranking.

    Example:
        >>> logits = torch.tensor([[0, 1, 0]])
        >>> target = torch.tensor([[1]])
        >>> accuracy(logits, target)
        [tensor(100.)]

    """
    KORNIA_CHECK_IS_TENSOR(pred)
    KORNIA_CHECK_IS_TENSOR(target)
    KORNIA_CHECK_SAME_DEVICE(pred, target)
    KORNIA_CHECK_SHAPE(pred, ["B", "C"])
    # one class index per sample; size-1 dimensions after the first, as in (B, 1), are read the same way
    KORNIA_CHECK_SIZES(target, [pred.shape[0], *([1] * (target.ndim - 1))])
    KORNIA_CHECK(
        len(topk) > 0 and all(_is_positive_index(k) for k in topk),
        "topk must be a non-empty sequence of positive integers",
    )

    maxk = min(max(topk), pred.size()[1])
    batch_size = target.size(0)
    _, pred = pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]
