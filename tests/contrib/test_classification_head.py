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

import pytest
import torch

import kornia

from testing.base import BaseTester


class TestClassificationHead(BaseTester):
    @pytest.mark.parametrize("B, D, N", [(1, 8, 10), (2, 2, 5)])
    def test_smoke(self, device, dtype, B, D, N):
        feat = torch.rand(B, D, D, device=device, dtype=dtype)
        head = kornia.contrib.ClassificationHead(embed_size=D, num_classes=N).to(device, dtype)
        logits = head(feat)
        assert logits.shape == (B, N)
