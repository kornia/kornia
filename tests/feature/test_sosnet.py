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

from kornia.feature import SOSNet

from testing.base import BaseTester


class TestSOSNet(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        sosnet = SOSNet(pretrained=False).to(device)
        sosnet.eval()  # batchnorm with size 1 is not allowed in train mode
        out = sosnet(inp)
        assert out.shape == (1, 128)

    def test_shape_batch(self, device):
        inp = torch.ones(16, 1, 32, 32, device=device)
        sosnet = SOSNet(pretrained=False).to(device)
        out = sosnet(inp)
        assert out.shape == (16, 128)

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=torch.float64)
        sosnet = SOSNet(pretrained=False).to(patches.device, patches.dtype)
        self.gradcheck(sosnet, (patches,), eps=1e-4, atol=1e-4)

    @pytest.mark.parametrize("patch_value", [0.0, 0.5])
    def test_degenerate_patch_gives_finite_descriptors(self, device, dtype, patch_value):
        # `forward`'s `eps` is what keeps `desc_norm`'s division defined for a patch the network maps
        # to exactly zero -- every `Conv2d` here has `bias=False` and every `BatchNorm2d` is
        # `affine=False` with `running_mean == 0`, so a constant patch reaches the normalisation as
        # exactly zero. 1e-10 flushes to 0.0 in float16, so the guard used to be inert there and every
        # descriptor came back NaN (kornia#4224).
        patches = torch.full((2, 1, 32, 32), patch_value, device=device, dtype=dtype)
        sosnet = SOSNet().to(device, dtype).eval()
        descriptors = sosnet(patches)
        assert torch.isfinite(descriptors).all(), f"{int(descriptors.isnan().sum())} non-finite descriptor entries"

    def test_dynamo(self, device, dtype, torch_optimizer):
        # The normalisation branches on `dtype` to lift half precision into float32. dtype is static
        # metadata, so dynamo resolves that branch at trace time; `torch_optimizer` does not pass
        # `fullgraph=True`, so what this pins is that the compiled result still matches eager on every
        # dtype -- including the lifted half path, which returns NaN if the branch is dropped. The
        # graph-break count is checked separately: `torch._dynamo.explain` reports 1 graph and 0 breaks.
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=dtype)
        model = SOSNet().to(device, dtype).eval()
        op = torch_optimizer(model)
        if device.type == "cuda" and dtype == torch.float32:
            # cuDNN convolutions keep PyTorch's TF32 default -- `conftest.py` only sets
            # `set_float32_matmul_precision`, never `torch.backends.cudnn.allow_tf32` -- so eager and
            # inductor pick different convolution algorithms and this comparison would measure TF32
            # conv reproducibility across seven convolutions rather than the branch surviving capture.
            # TF32 carries a 10-bit mantissa, so ~1e-3 at the activations' unit scale; the worst
            # observed over five seeds on an RTX 4090 is 2.8e-4.
            self.assert_close(op(patches), model(patches), atol=1e-3, rtol=1e-3)
        else:
            self.assert_close(op(patches), model(patches))

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        # `torch.ones` drives a randomly initialised SOSNet to exactly zero (a ReLU collapse), which made
        # this comparison vacuous: both descriptors came out as the constant 1/sqrt(128) and the test
        # scripted a *second*, independently initialised model without noticing. `torch.rand` keeps the
        # network's output non-degenerate; scripting the same instance is what makes the two sides
        # comparable at all. See TestHyNet::test_jit for the pattern.
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = SOSNet().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(model)
        self.assert_close(model(patches), model_jit(patches))
