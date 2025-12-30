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

from kornia.models.vit import VisionTransformer

from testing.base import BaseTester


class TestVisionTransformer(BaseTester):
    @pytest.mark.parametrize("B", [1, 2])
    @pytest.mark.parametrize("H", [1, 3, 8])
    @pytest.mark.parametrize("D", [240, 768])
    @pytest.mark.parametrize("image_size", [32, 224])
    def test_smoke(self, device, dtype, B, H, D, image_size):
        patch_size = 16
        T = image_size**2 // patch_size**2 + 1  # tokens size

        img = torch.rand(B, 3, image_size, image_size, device=device, dtype=dtype)
        vit = VisionTransformer(image_size=image_size, num_heads=H, embed_dim=D).to(device, dtype)

        out = vit(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (B, T, D)

        feats = vit.encoder_results
        assert isinstance(feats, list)
        assert len(feats) == 12
        for f in feats:
            assert f.shape == (B, T, D)

    @pytest.mark.parametrize("H", [3, 8])
    @pytest.mark.parametrize("D", [245, 1001])
    @pytest.mark.parametrize("image_size", [32, 224])
    def test_exception(self, device, dtype, H, D, image_size):
        with pytest.raises(ValueError):
            VisionTransformer(image_size=image_size, num_heads=H, embed_dim=D).to(device, dtype)

    def test_backbone(self, device, dtype):
        def backbone_mock(x):
            return torch.ones(1, 128, 14, 14, device=device, dtype=dtype)

        img = torch.rand(1, 3, 32, 32, device=device, dtype=dtype)
        vit = VisionTransformer(backbone=backbone_mock, num_heads=8).to(device, dtype)
        out = vit(img)
        assert out.shape == (1, 197, 128)
