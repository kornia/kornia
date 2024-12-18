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

from kornia.feature import TFeat

from testing.base import BaseTester


class TestTFeat(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        tfeat = TFeat().to(device)
        tfeat.eval()  # batchnorm with size 1 is not allowed in train mode
        out = tfeat(inp)
        assert out.shape == (1, 128)

    @pytest.mark.slow
    def test_pretrained(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        tfeat = TFeat(True).to(device)
        tfeat.eval()  # batchnorm with size 1 is not allowed in train mode
        out = tfeat(inp)
        assert out.shape == (1, 128)

    def test_shape_batch(self, device):
        inp = torch.ones(16, 1, 32, 32, device=device)
        tfeat = TFeat().to(device)
        out = tfeat(inp)
        assert out.shape == (16, 128)

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=torch.float64)
        tfeat = TFeat().to(patches.device, patches.dtype)
        self.gradcheck(tfeat, (patches,), eps=1e-2, atol=1e-2)

    @pytest.mark.slow
    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        tfeat = TFeat(True).to(patches.device, patches.dtype).eval()
        tfeat_jit = torch.jit.script(TFeat(True).to(patches.device, patches.dtype).eval())
        self.assert_close(tfeat_jit(patches), tfeat(patches))
