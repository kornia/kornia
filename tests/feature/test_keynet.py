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

from kornia.feature import KeyNet

from testing.base import BaseTester


class TestKeyNet(BaseTester):
    def test_shape(self, device, dtype):
        inp = torch.rand(1, 1, 16, 16, device=device, dtype=dtype)
        keynet = KeyNet().to(device, dtype)
        out = keynet(inp)
        assert out.shape == inp.shape

    def test_shape_batch(self, device, dtype):
        inp = torch.ones(16, 1, 16, 16, device=device, dtype=dtype)
        keynet = KeyNet().to(device, dtype)
        out = keynet(inp)
        assert out.shape == inp.shape

    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 16, 16, device=device, dtype=torch.float64)
        keynet = KeyNet().to(patches.device, patches.dtype)
        self.gradcheck(keynet, (patches,), eps=1e-4, atol=1e-4, nondet_tol=1e-8)
