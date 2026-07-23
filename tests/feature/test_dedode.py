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

from kornia.feature.dedode import DeDoDe
from kornia.feature.dedode.transformer.layers.block import Block


class TestDeDoDeBlock:
    def test_drop_path2_applied_to_ffn_residual(self, device, dtype):
        # Regression test: `Block.forward` must route the attention residual through
        # `drop_path1` and the FFN residual through `drop_path2`. A prior copy-paste
        # bug applied `drop_path1` to both, silently ignoring `drop_path2`.
        block = Block(dim=8, num_heads=2, drop_path=0.05).to(device, dtype)
        block.train()

        calls = {"drop_path1": 0, "drop_path2": 0}
        orig_dp1, orig_dp2 = block.drop_path1.forward, block.drop_path2.forward

        def spy_dp1(x):
            calls["drop_path1"] += 1
            return orig_dp1(x)

        def spy_dp2(x):
            calls["drop_path2"] += 1
            return orig_dp2(x)

        block.drop_path1.forward = spy_dp1
        block.drop_path2.forward = spy_dp2

        x = torch.randn(2, 4, 8, device=device, dtype=dtype)
        block(x)

        assert calls == {"drop_path1": 1, "drop_path2": 1}


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
