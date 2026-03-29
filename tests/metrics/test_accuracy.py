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

from kornia.metrics import accuracy


class TestAccuracy:
    def test_top1_perfect(self):
        logits = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        target = torch.tensor([[1], [0]])
        result = accuracy(logits, target, topk=(1,))
        assert len(result) == 1
        assert result[0].item() == 100.0

    def test_top1_zero(self):
        logits = torch.tensor([[1.0, 0.0, 0.0]])
        target = torch.tensor([[2]])
        result = accuracy(logits, target, topk=(1,))
        assert result[0].item() == 0.0

    def test_topk(self):
        # 3 classes; true class is index 2 (lowest logit) — wrong for both top-1 and top-2
        logits = torch.tensor([[0.3, 0.5, 0.2]])
        target = torch.tensor([[2]])  # true class is index 2 (third highest)
        top1, top2 = accuracy(logits, target, topk=(1, 2))
        assert top1.item() == 0.0
        assert top2.item() == 0.0

        logits2 = torch.tensor([[0.3, 0.2, 0.5]])
        target2 = torch.tensor([[2]])
        top1b, _ = accuracy(logits2, target2, topk=(1, 2))
        assert top1b.item() == 100.0

    def test_topk_exceeds_num_classes_is_clipped(self):
        # topk=5 but only 3 classes — should not crash
        logits = torch.tensor([[0.1, 0.8, 0.1]])
        target = torch.tensor([[1]])
        result = accuracy(logits, target, topk=(5,))
        assert result[0].item() == 100.0
