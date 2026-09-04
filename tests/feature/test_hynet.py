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

    @pytest.mark.parametrize("patch_value", [0.0, 0.5])
    @pytest.mark.parametrize("is_bias", [True, False])
    def test_degenerate_patch_gives_finite_descriptors(self, device, dtype, patch_value, is_bias):
        # `eps_l2_norm` is what keeps `desc_norm`'s division defined for a patch the network maps to
        # exactly zero, and 1e-10 flushes to 0.0 in float16, so the guard used to be inert there and
        # every descriptor came back NaN (kornia#4224). `is_bias=False` is the configuration that
        # reaches it: with the TLU bias on, the pre-normalisation tensor never lands exactly on zero.
        patches = torch.full((2, 1, 32, 32), patch_value, device=device, dtype=dtype)
        hynet = HyNet(is_bias=is_bias).to(device, dtype).eval()
        descriptors = hynet(patches)
        assert torch.isfinite(descriptors).all(), f"{int(descriptors.isnan().sum())} non-finite descriptor entries"

    def test_dynamo(self, device, dtype, torch_optimizer):
        # The normalisation branches on `dtype` to lift half precision into float32. dtype is static
        # metadata, so dynamo resolves that branch at trace time; `torch_optimizer` does not pass
        # `fullgraph=True`, so what this pins is that the compiled result still matches eager on every
        # dtype -- including the lifted half path, which returns NaN if the branch is dropped. The
        # graph-break count is checked separately: `torch._dynamo.explain` reports 1 graph and 0 breaks.
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=dtype)
        model = HyNet().to(device, dtype).eval()
        op = torch_optimizer(model)
        self.assert_close(op(patches), model(patches))

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = HyNet().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(model)
        self.assert_close(model(patches), model_jit(patches))
