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

from kornia.feature import HardNet, HardNet8

from testing.base import BaseTester, supports_conv2d


class TestHardNet(BaseTester):
    @pytest.mark.slow
    def test_shape(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        hardnet = HardNet().to(device)
        hardnet.eval()  # batchnorm with size 1 is not allowed in train mode
        out = hardnet(inp)
        assert out.shape == (1, 128)

    @pytest.mark.slow
    def test_shape_batch(self, device):
        inp = torch.ones(4, 1, 32, 32, device=device)
        hardnet = HardNet().to(device)
        out = hardnet(inp)
        assert out.shape == (4, 128)

    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=torch.float64)
        hardnet = HardNet().to(patches.device, patches.dtype)
        self.gradcheck(hardnet, (patches,), eps=1e-4, atol=1e-4, nondet_tol=1e-8)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = HardNet().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(HardNet().to(patches.device, patches.dtype).eval())
        self.assert_close(model(patches), model_jit(patches))


class TestHardNet8(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        hardnet = HardNet8().to(device)
        hardnet.eval()  # batchnorm with size 1 is not allowed in train mode
        out = hardnet(inp)
        assert out.shape == (1, 128)

    def test_shape_batch(self, device):
        inp = torch.ones(4, 1, 32, 32, device=device)
        hardnet = HardNet8().to(device)
        out = hardnet(inp)
        assert out.shape == (4, 128)

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=torch.float32)
        hardnet = HardNet8().to(patches.device, patches.dtype)
        self.gradcheck(hardnet, (patches,), eps=1e-4, atol=1e-4)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = HardNet8().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(HardNet8().to(patches.device, patches.dtype).eval())
        self.assert_close(model(patches), model_jit(patches))

    def test_untrained_descriptors_are_distinct_5692(self, device, dtype):
        # An all-ones PCA placeholder mapped every patch to +-(1, ..., 1) / sqrt(128).
        if not supports_conv2d(device, dtype):
            pytest.skip(f"no conv2d kernel for {dtype} on {device.type}")
        torch.manual_seed(0)
        patches = torch.rand(8, 1, 32, 32, device=device, dtype=dtype)
        out = HardNet8().to(device, dtype)(patches)
        assert out.shape == (8, 128)
        self.assert_close(out.norm(dim=1), torch.ones(8, device=device, dtype=dtype))
        cos = out @ out.T
        off_diagonal = cos[~torch.eye(8, dtype=torch.bool, device=device)]
        assert off_diagonal.abs().max() < 0.99

    def test_untrained_model_keeps_the_first_features_5692(self, device, dtype):
        if not supports_conv2d(device, dtype):
            pytest.skip(f"no conv2d kernel for {dtype} on {device.type}")
        torch.manual_seed(0)
        patches = torch.rand(4, 1, 32, 32, device=device, dtype=dtype)
        model = HardNet8().to(device, dtype)
        features = model.features(model._normalize_input(patches)).view(4, -1)
        expected = torch.nn.functional.normalize(torch.nn.functional.normalize(features, dim=1)[:, :128], dim=1)
        self.assert_close(model(patches), expected)

    def test_untrained_model_is_trainable_5692(self, device, dtype):
        # The all-ones placeholder made the normalised output constant, so the gradient was zero (~1e-15).
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("half-precision rounding leaves a noise gradient on the degenerate placeholder as well")
        torch.manual_seed(0)
        patches = torch.rand(4, 1, 32, 32, device=device, dtype=dtype)
        model = HardNet8().to(device, dtype)
        target = torch.randn(4, 128, device=device, dtype=dtype)
        (model(patches) * target).sum().backward()
        grad = model.features[0].weight.grad
        assert grad is not None
        assert grad.abs().max() > 1e-3

    def test_checkpoint_layout_is_unchanged_5692(self, device):
        # The pretrained checkpoint loads with strict=True, so the buffer names and shapes must stay.
        model = HardNet8().to(device)
        state = model.state_dict()
        assert state["components"].shape == (512, 128)
        assert state["mean"].shape == (512,)
        loaded = torch.rand(512, 128, device=device)
        state["components"] = loaded
        model.load_state_dict(state, strict=True)
        self.assert_close(model.components, loaded)


class TestHardNetConstantPatchIsFinite(BaseTester):
    """A constant patch drives the features to zero; the L2 normalisation must not give NaN.

    `F.normalize`'s default eps of 1e-12 rounds to zero in float16, so `0 / 0` reached the
    descriptor. A detector that pads a short result feeds exactly this patch through a zero LAF.
    """

    @pytest.mark.parametrize("desc_dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("model", [HardNet, HardNet8])
    def test_constant_patch(self, device, desc_dtype, model):
        if not supports_conv2d(device, desc_dtype):
            # The descriptor is a stack of convolutions; torch 2.1.2 has no float16 CPU kernel.
            pytest.skip(f"no conv2d kernel for {desc_dtype} on {device.type}")
        patches = torch.zeros(2, 1, 32, 32, device=device, dtype=desc_dtype)
        out = model().to(device, desc_dtype)(patches)
        assert torch.isfinite(out).all()
