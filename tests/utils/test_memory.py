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

import kornia

from testing.base import BaseTester


class TestBatchedForward(BaseTester):
    def test_runbatch(self, device):
        patches = torch.rand(34, 1, 32, 32)
        sift = kornia.feature.SIFTDescriptor(32)
        desc_batched = kornia.utils.memory.batched_forward(sift, patches, device, 32)
        desc = sift(patches)
        assert torch.allclose(desc, desc_batched)

    def test_runone(self, device):
        patches = torch.rand(16, 1, 32, 32)
        sift = kornia.feature.SIFTDescriptor(32)
        desc_batched = kornia.utils.memory.batched_forward(sift, patches, device, 32)
        desc = sift(patches)
        assert torch.allclose(desc, desc_batched)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(
            kornia.utils.memory.batched_forward, (kornia.feature.BlobHessian(), img, device, 2), nondet_tol=1e-4
        )
