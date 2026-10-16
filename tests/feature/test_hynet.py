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

import math

import pytest
import torch

from kornia.core.check import ShapeError
from kornia.feature import TLU, FilterResponseNorm2d, HyNet

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available, supports_conv2d


class TestHyNet(BaseTester):
    def test_learnable_eps(self, device):
        layer = FilterResponseNorm2d(4, is_eps_leanable=True).to(device)
        output = layer(torch.ones(2, 4, 8, 8, device=device))
        assert output.shape == (2, 4, 8, 8)
        assert layer.eps.requires_grad

    def test_shape(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        hynet = HyNet(pretrained=False).to(device).eval()
        out = hynet(inp)
        assert out.shape == (1, 128)

    @pytest.mark.parametrize("shape", [(2, 1, 40, 40), (2, 1, 32, 31), (2, 3, 32, 32), (1, 32, 32)])
    def test_wrong_patch_shape_is_rejected_5724(self, device, shape):
        # without the check a 40x40 patch silently produced a (2, 1152) descriptor
        with pytest.raises(ShapeError):
            HyNet(pretrained=False).to(device).eval()(torch.ones(*shape, device=device))

    def test_shape_batch(self, device):
        inp = torch.ones(4, 1, 32, 32, device=device)
        hynet = HyNet(pretrained=False).to(device).eval()
        out = hynet(inp)
        assert out.shape == (4, 128)

    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=torch.float64)
        hynet = HyNet(pretrained=False).to(patches.device, patches.dtype).eval()
        self.gradcheck(hynet, (patches,), eps=1e-4, atol=1e-4, nondet_tol=1e-8)

    @pytest.mark.parametrize("patch_value", [0.0, 0.5])
    @pytest.mark.parametrize("is_bias", [True, False])
    def test_degenerate_patch_gives_finite_descriptors(self, device, dtype, patch_value, is_bias):
        # `eps_l2_norm` is what keeps `desc_norm`'s division defined for a patch the network maps to
        # exactly zero, and `is_bias=False` is the configuration that gets there: with the TLU bias on,
        # the pre-normalisation tensor never lands exactly on zero. 1e-10 flushes to 0.0 in float16, so
        # the guard used to be inert and every descriptor came back NaN (kornia#4224). bfloat16 keeps
        # float32's exponent range and never had that failure, but on CPU both half dtypes used to
        # raise out of `avg_pool3d` before reaching the division at all, so both are pinned here.
        patches = torch.full((2, 1, 32, 32), patch_value, device=device, dtype=dtype)
        hynet = HyNet(pretrained=False, is_bias=is_bias).to(device, dtype).eval()
        descriptors = hynet(patches)
        assert torch.isfinite(descriptors).all(), (
            f"{int((~torch.isfinite(descriptors)).sum())} non-finite descriptor entries"
        )

    def test_dynamo(self, device, dtype, torch_optimizer, cudnn_tf32_follows_option):
        # The normalisation branches on `dtype` to lift half precision into float32. dtype is static
        # metadata, so dynamo resolves that branch at trace time; what this pins is that the compiled
        # result still matches eager on every dtype -- including the lifted half path, which returns
        # NaN if the branch is dropped. `cudnn_tf32_follows_option` is what makes the float32 leg
        # meaningful on CUDA; see its docstring.
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=dtype)
        model = HyNet(pretrained=False).to(device, dtype).eval()
        op = torch_optimizer(model)
        self.assert_close(op(patches), model(patches))

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_dynamo_fullgraph(self, device, dtype, cudnn_tf32_follows_option):
        # `export-support.rst` publishes HyNet as capturable with zero graph breaks. The dtype branch
        # reads static metadata and must not become the thing that breaks the graph.
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=dtype)
        model = HyNet(pretrained=False).to(device, dtype).eval()
        expected = model(patches)
        torch._dynamo.reset()
        compiled = torch.compile(model, fullgraph=True)
        self.assert_close(compiled(patches), expected)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = HyNet(pretrained=False).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(model)
        self.assert_close(model(patches), model_jit(patches))


class TestConventionsHyNet(BaseTester):
    """Pins for HyNet's input-range rule and its two normalisation layers, FilterResponseNorm2d and TLU."""

    def test_convention_hynet_is_scale_invariant_but_not_offset_invariant(self, device, dtype):
        # HyNet's first layer is FilterResponseNorm2d on the raw patch: it divides by the RMS and subtracts no mean, as
        # in the reference (yuruntian/HyNet model.py, trained on uint8 patches cast to float). So 255x gives x's
        # descriptor, while x + 1 and the mean-subtracted x - 0.5 do not (best row cosine about 0.9 and 0.7 with random
        # weights). Feed it non-negative intensities, not standardised patches.
        if not supports_conv2d(device, dtype):
            pytest.skip(f"no conv2d kernel for {dtype} on {device.type}")
        torch.manual_seed(0)
        model = HyNet(pretrained=False).to(device, dtype).eval()
        patches = torch.rand(4, 1, 32, 32, device=device, dtype=dtype)
        desc = model(patches)
        self.assert_close(model(255 * patches), desc)
        for moved in (patches + 1, patches - 0.5):
            cos = torch.nn.functional.cosine_similarity(model(moved).float(), desc.float(), dim=1)
            assert cos.max() < 0.95, cos

    def test_convention_filter_response_norm_divides_each_map_by_its_rms_with_eps_under_the_root(self, device, dtype):
        # y = weight * x / sqrt(mean_HW(x^2) + |eps|) + bias, per sample and channel (weight 1, bias 0 at init).
        # eps = 1 makes its place visible: a constant map of 1 gives 1 / sqrt(2), where eps outside the root would give
        # 1 / (1 + 1) = 0.5; a constant 3 gives 3 / sqrt(10), not 0.75.
        frn = FilterResponseNorm2d(2, eps=1.0).to(device, dtype)
        x = torch.tensor(
            [[[[1.0, 1.0, 1.0, 1.0]], [[3.0, 3.0, 3.0, 3.0]]], [[[0.0, 1.0, 2.0, 3.0]], [[-2.0, 0.0, 0.0, 2.0]]]],
            device=device,
            dtype=dtype,
        )
        # mean squares 1, 9, 3.5 and 2, each plus eps
        expected = torch.tensor(
            [
                [[[1 / math.sqrt(2)] * 4], [[3 / math.sqrt(10)] * 4]],
                [
                    [[0.0, 1 / math.sqrt(4.5), 2 / math.sqrt(4.5), 3 / math.sqrt(4.5)]],
                    [[-2 / math.sqrt(3), 0, 0, 2 / math.sqrt(3)]],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(frn(x), expected)
        self.assert_close(frn(x[1:]), expected[1:])  # each sample on its own: no batch statistics
        self.assert_close(FilterResponseNorm2d(2, eps=-1.0).to(device, dtype)(x), expected)  # |eps|

    def test_convention_tlu_floors_each_channel_at_its_tau_initialised_to_minus_one(self, device, dtype):
        tlu = TLU(3).to(device, dtype)
        assert tlu.tau.shape == (1, 3, 1, 1)
        x = torch.tensor([-3.0, -1.0, -0.5, 2.0], device=device, dtype=dtype).view(1, 1, 1, 4).repeat(2, 3, 1, 1)
        at_minus_one = torch.tensor([-1.0, -1.0, -0.5, 2.0], device=device, dtype=dtype)
        self.assert_close(tlu(x), at_minus_one.expand(2, 3, 1, 4))
        with torch.no_grad():
            tlu.tau[0, 1] = 0.0
        # max(x, tau) per channel: only channel 1 now floors at 0
        at_zero = torch.tensor([0.0, 0.0, 0.0, 2.0], device=device, dtype=dtype)
        self.assert_close(
            tlu(x), torch.stack([at_minus_one, at_zero, at_minus_one]).view(1, 3, 1, 4).expand(2, 3, 1, 4)
        )
