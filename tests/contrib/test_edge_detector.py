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

import kornia

from testing.base import BaseTester


class TestEdgeDetector(BaseTester):
    @pytest.mark.slow
    def test_smoke(self, device, dtype):
        img = torch.rand(2, 3, 64, 64, device=device, dtype=dtype)
        net = kornia.contrib.EdgeDetectorBuilder.build(pretrained=False).to(device, dtype)
        out = net(img)
        # ResizePostProcessor returns a list, so we need to handle that
        if isinstance(out, list):
            assert len(out) == 2
            assert all(item.shape == (1, 1, 64, 64) for item in out)
        else:
            assert out.shape == (2, 1, 64, 64)

    @pytest.mark.slow
    @pytest.mark.skip(reason="issue with `ClassVar[list[int]]`")
    def test_jit(self, device, dtype):
        op = kornia.contrib.EdgeDetectorBuilder.build(pretrained=False).to(device, dtype)
        op_jit = torch.jit.script(op)
        assert op_jit is not None
