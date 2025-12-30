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

from kornia.models import DexiNed

from testing.base import BaseTester


class TestDexiNed(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(2, 3, 32, 32, device=device, dtype=dtype)
        net = DexiNed(pretrained=False).to(device, dtype)
        feat = net.get_features(img)
        assert len(feat) == 6
        out = net(img)
        assert out.shape == (2, 1, 32, 32)

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["dexined"], indirect=True)
    def test_inference(self, device, dtype, data):
        model = DexiNed(pretrained=False)
        model.load_state_dict(data, strict=True)
        model = model.to(device, dtype)
        model.eval()

        img = torch.tensor([[[[0.0, 255.0, 0.0], [0.0, 255.0, 0.0], [0.0, 255.0, 0.0]]]], device=device, dtype=dtype)
        img = img.repeat(1, 3, 1, 1)

        expect = torch.tensor(
            [[[[-0.3709, 0.0519, -0.2839], [0.0627, 0.6587, -0.1276], [-0.1840, -0.3917, -0.8240]]]],
            device=device,
            dtype=dtype,
        )

        out = model(img)
        self.assert_close(out, expect, atol=3e-4, rtol=3e-4)

    @pytest.mark.skip(reason="DexiNed do not compile with dynamo.")
    def test_dynamo(self, device, dtype, torch_optimizer):
        # TODO: update the dexined to be possible to use with dynamo
        data = torch.rand(2, 3, 32, 32, device=device, dtype=dtype)
        op = DexiNed(pretrained=True).to(device, dtype)
        op_optimized = torch_optimizer(op)

        expected = op(data)
        actual = op_optimized(data)

        self.assert_close(actual, expected)
