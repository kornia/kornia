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

from kornia.feature import FilterResponseNorm2d, HyNet

from testing.base import BaseTester


class TestHyNet(BaseTester):
    def test_learnable_eps(self, device):
        layer = FilterResponseNorm2d(4, is_eps_leanable=True).to(device)
        output = layer(torch.ones(2, 4, 8, 8, device=device))
        assert output.shape == (2, 4, 8, 8)
        assert layer.eps.requires_grad

    def test_shape(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        hynet = HyNet().to(device)
        out = hynet(inp)
        assert out.shape == (1, 128)

    def test_shape_batch(self, device):
        inp = torch.ones(4, 1, 32, 32, device=device)
        hynet = HyNet().to(device)
        out = hynet(inp)
        assert out.shape == (4, 128)

    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=torch.float64)
        hynet = HyNet().to(patches.device, patches.dtype)
        self.gradcheck(hynet, (patches,), eps=1e-4, atol=1e-4, nondet_tol=1e-8)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = HyNet().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(model)
        self.assert_close(model(patches), model_jit(patches))
