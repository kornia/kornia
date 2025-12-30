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

from kornia.models import MobileViT

from testing.base import BaseTester


class TestMobileViT(BaseTester):
    @pytest.mark.parametrize("B", [1, 2])
    @pytest.mark.parametrize("image_size", [(256, 256)])
    @pytest.mark.parametrize("mode", ["xxs", "xs", "s"])
    @pytest.mark.parametrize("patch_size", [(2, 2)])
    def test_smoke(self, device, dtype, B, image_size, mode, patch_size):
        ih, iw = image_size
        channel = {"xxs": 320, "xs": 384, "s": 640}

        img = torch.rand(B, 3, ih, iw, device=device, dtype=dtype)
        mvit = MobileViT(mode=mode, patch_size=patch_size).to(device, dtype)

        out = mvit(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (B, channel[mode], 8, 8)
