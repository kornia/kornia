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

import warnings

import pytest
import torch

from kornia.feature.dedode import DeDoDe
from kornia.feature.dedode.decoder import ConvRefiner
from kornia.feature.dedode.transformer.layers import MemEffAttention, NestedTensorBlock

from testing.base import BaseTester


class TestConvRefiner(BaseTester):
    def test_amp_matches_input_device(self, device):
        refiner = ConvRefiner(in_dim=4, hidden_dim=4, out_dim=4).to(device)
        autocast_enabled = []
        # ``torch.is_autocast_enabled`` only accepts a device type from torch 2.4 onwards; the
        # no-argument spelling reports the CUDA state on every torch kornia supports, which is
        # the state the ``torch.autocast("cuda", ...)`` region inside ``ConvRefiner`` controls.
        refiner.block1.register_forward_pre_hook(
            lambda _module, _inputs: autocast_enabled.append(torch.is_autocast_enabled())
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            refiner(torch.rand(1, 4, 8, 8, device=device))

        assert autocast_enabled == [device.type == "cuda"]
        assert not any("device_type of 'cuda'" in str(w.message) for w in caught)


class TestMemEffAttention(BaseTester):
    def test_attn_bias_raises(self, device, dtype):
        attention = MemEffAttention(dim=8, num_heads=2).to(device, dtype).eval()
        x = torch.rand(1, 5, 8, device=device, dtype=dtype)

        with pytest.raises(NotImplementedError, match="attn_bias is not supported"):
            attention(x, attn_bias=torch.zeros(1, 2, 5, 5, device=device, dtype=dtype))


class TestNestedTensorBlock(BaseTester):
    def test_tensor_input(self, device, dtype):
        block = NestedTensorBlock(dim=8, num_heads=2).to(device, dtype).eval()
        x = torch.rand(1, 5, 8, device=device, dtype=dtype)

        out = block(x)

        assert out.shape == (1, 5, 8)
        # the block is a residual one, so the output has to differ from the input by the
        # attention and feed-forward branches actually having run.
        assert not torch.allclose(out, x)

    def test_list_input_raises(self, device, dtype):
        block = NestedTensorBlock(dim=8, num_heads=2).to(device, dtype).eval()
        x = torch.rand(1, 5, 8, device=device, dtype=dtype)

        with pytest.raises(TypeError, match="nested-tensor list path was removed"):
            block([x])


@pytest.mark.skip(reason="DeDoDe is ummaintained")
class TestDeDoDe:
    @pytest.mark.slow
    @pytest.mark.parametrize("descriptor_model", ["B", "G"])
    @pytest.mark.parametrize("detector_model", ["L"])
    def test_smoke(self, dtype, device, descriptor_model, detector_model):
        if "G" in descriptor_model and device.type != "cuda" and dtype == torch.float16:
            pytest.skip('G descriptors do not support no cuda device. "LayerNormKernelImpl" not implemented for `Half`')
        dedode = DeDoDe(descriptor_model=descriptor_model, detector_model=detector_model, amp_dtype=dtype).to(
            device, dtype
        )
        shape = (2, 3, 128, 128)
        n = 1000
        inp = torch.randn(*shape, device=device, dtype=dtype)
        keypoints, scores, descriptions = dedode(inp, n=n)
        assert keypoints.shape == (shape[0], n, 2)
        assert scores.shape == (shape[0], n)
        assert descriptions.shape == (shape[0], n, 256)
