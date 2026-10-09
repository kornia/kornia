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

from typing import List, Tuple

import torch


def accuracy(pred: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """Compute the accuracy over the k top predictions for the specified values of k.

    Convention:
        - ``pred`` holds one score per class, :math:`(B, C)`; only the order of the scores in a row matters.
          ``target`` holds class indices, :math:`(B,)` or :math:`(B, 1)`.
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
    maxk = min(max(topk), pred.size()[1])
    batch_size = target.size(0)
    _, pred = pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]
