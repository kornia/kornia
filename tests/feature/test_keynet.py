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
    def test_dynamo(self, device, dtype, torch_optimizer):
        inp = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        model = KeyNet().to(device, dtype)
        self.assert_close(torch_optimizer(model)(inp), model(inp))

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


class TestConventionsKeyNet(BaseTester):
    def test_convention_keynet_response_is_non_negative_at_the_input_size(self, device, dtype):
        # KeyNet ends in a ReLU and resamples every pyramid level's features back to the input size, so the response is
        # a non-negative (B, 1, H, W) map aligned with the input, also for an odd, non-square size. With the random
        # weights of seed 0 and the last bias set to -0.01, the last pre-activation spans [-0.014, 0.013] (CPU float64),
        # so the ReLU clips about a tenth of the map and the bound below is not met by an all-zero map.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = KeyNet()
        with torch.no_grad():
            model.last_conv[0].bias.fill_(-0.01)
        img = torch.rand(2, 1, 37, 53, generator=torch.Generator().manual_seed(0)).to(device, dtype)
        out = model.to(device, dtype)(img)
        assert out.shape == (2, 1, 37, 53)
        assert bool((out >= 0).all())
        assert bool((out == 0).any())
        assert bool((out > 0).any())
