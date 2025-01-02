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

from kornia.core import Tensor


def accuracy(pred: Tensor, target: Tensor, topk: Tuple[int, ...] = (1,)) -> List[Tensor]:
    """Compute the accuracy over the k top predictions for the specified values of k.

    Args:
        pred: the input tensor with the logits to evaluate.
        target: the tensor containing the ground truth.
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
