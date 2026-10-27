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
from torch import nn

from kornia.models.vit import VisionTransformer

from testing.base import BaseTester


class TestVisionTransformer(BaseTester):
    @pytest.mark.parametrize(
        ("B", "H", "D", "image_size"),
        [
            (2, 1, 240, 32),
            (1, 1, 768, 224),
            (2, 3, 768, 32),
            (1, 3, 240, 224),
            (2, 8, 240, 224),
            (1, 8, 768, 32),
        ],
    )
    def test_smoke(self, device, dtype, B, H, D, image_size):
        patch_size = 16
        depth = 2
        T = image_size**2 // patch_size**2 + 1  # tokens size

        img = torch.rand(B, 3, image_size, image_size, device=device, dtype=dtype)
        vit = VisionTransformer(image_size=image_size, num_heads=H, embed_dim=D, depth=depth).to(device, dtype)

        out = vit(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (B, T, D)

        feats = vit.encoder_results
        assert isinstance(feats, list)
        assert len(feats) == depth
        for f in feats:
            assert f.shape == (B, T, D)

    def test_empty_batch_6266(self, device, dtype):
        """An empty batch keeps the token and embedding dimensions."""
        vit = VisionTransformer(image_size=32, num_heads=2, embed_dim=64, depth=1).to(device, dtype)
        out = vit(torch.rand(0, 3, 32, 32, device=device, dtype=dtype))
        assert out.shape == (0, 5, 64)
        assert [f.shape for f in vit.encoder_results] == [(0, 5, 64)]

    @pytest.mark.parametrize("H", [3, 8])
    @pytest.mark.parametrize("D", [245, 1001])
    @pytest.mark.parametrize("image_size", [32, 224])
    def test_exception(self, device, dtype, H, D, image_size):
        with pytest.raises(ValueError):
            VisionTransformer(image_size=image_size, num_heads=H, embed_dim=D).to(device, dtype)

    @pytest.mark.parametrize(
        "shape",
        [(1, 3, 112, 448), (1, 3, 224, 239), (1, 3, 239, 224), (1, 3, 224, 240), (1, 3, 240, 240), (1, 1, 224, 224)],
        ids=["7x28-grid", "224x239", "239x224", "224x240", "240x240", "one-channel"],
    )
    def test_forward_rejects_other_sizes(self, device, dtype, shape):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        vit = VisionTransformer(image_size=224, patch_size=16, depth=1).to(device, dtype)
        assert vit(torch.rand(1, 3, 224, 224, device=device, dtype=dtype)).shape == (1, 197, 768)

        # The position table holds (224 // 16) ** 2 + 1 = 197 rows. A 112 x 448 image has a 7 x 28 grid of
        # 196 = 14 * 14 patches, and 224 x 239 and 239 x 224 a 14 x 14 grid (the last 239 - 14 * 16 = 15 columns or
        # rows dropped), so these fit the table; 224 x 240 (14 x 15 + 1 = 211 tokens), 240 x 240 (15 x 15 + 1 = 226)
        # and one channel do not. 224 x 239 and 239 x 224 have one side right, so the check must test both.
        with pytest.raises(ValueError, match="224"):
            vit(torch.rand(shape, device=device, dtype=dtype))

    @pytest.mark.parametrize("with_backbone", [False, True], ids=["patches", "adaptive-backbone"])
    def test_dynamo_fullgraph(self, device, dtype, torch_optimizer, with_backbone):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        # The size and token-grid checks read static shapes: they must compile without a graph break. The adaptive
        # backbone gives its 2 x 2 grid for the 32 x 64 input as well.
        backbone = nn.Sequential(nn.Conv2d(3, 64, 16, 16), nn.AdaptiveAvgPool2d((2, 2))) if with_backbone else None
        vit = VisionTransformer(image_size=32, patch_size=16, embed_dim=64, depth=1, num_heads=2, backbone=backbone)
        vit = vit.to(device, dtype).eval()
        img = torch.rand(2, 3, 32, 64 if with_backbone else 32, device=device, dtype=dtype)

        vit_optimized = torch_optimizer(vit, fullgraph=True)

        # Inductor keeps fused half-precision intermediates in float32 where eager rounds every op: on the CPU in
        # bfloat16, one of the 640 outputs differs from eager by 0.031, two bfloat16 steps at its value of 2.9.
        low_tolerance = dtype in (torch.float16, torch.bfloat16)
        with torch.no_grad():
            self.assert_close(vit_optimized(img), vit(img), low_tolerance=low_tolerance)

    def test_backbone(self, device, dtype):
        def backbone_mock(x):
            return torch.ones(1, 128, 14, 14, device=device, dtype=dtype)

        # The mock gives its 14 x 14 grid, the one the position table is built for, whatever the input size.
        img = torch.rand(1, 3, 32, 32, device=device, dtype=dtype)
        vit = VisionTransformer(backbone=backbone_mock, num_heads=8).to(device, dtype)
        out = vit(img)
        assert out.shape == (1, 197, 128)
        assert len(vit.encoder_results) == 12

    @pytest.mark.parametrize("size", [(224, 224), (256, 256), (448, 448), (224, 448), (192, 320), (112, 448)])
    def test_adaptive_backbone_takes_other_sizes(self, device, dtype, size):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        if device.type == "mps" and any((side // 16) % 14 for side in size):
            pytest.skip("MPS adaptive pooling needs a grid divisible by its output size (pytorch/pytorch#96056)")
        # Adaptive pooling gives the 14 x 14 grid of a 224 input at every size, so the positions match the tokens.
        backbone = nn.Sequential(nn.Conv2d(3, 64, 16, 16), nn.AdaptiveAvgPool2d((14, 14)))
        vit = VisionTransformer(image_size=224, num_heads=4, depth=1, backbone=backbone).to(device, dtype)

        assert vit(torch.rand(1, 3, *size, device=device, dtype=dtype)).shape == (1, 197, 64)

    @pytest.mark.parametrize("size", [(112, 448), (240, 240)], ids=["7x28-grid", "15x15-grid"])
    def test_backbone_rejects_other_token_grids(self, device, dtype, size):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        # A stride-16 backbone gives the 14 x 14 grid only for a 224 input. 112 x 448 gives 7 x 28, 196 tokens that
        # fit the table with misplaced positions, and 240 x 240 gives 15 x 15.
        vit = VisionTransformer(image_size=224, num_heads=4, depth=1, backbone=nn.Conv2d(3, 64, 16, 16))
        vit = vit.to(device, dtype)
        assert vit(torch.rand(1, 3, 224, 224, device=device, dtype=dtype)).shape == (1, 197, 64)

        with pytest.raises(ValueError, match="14x14"):
            vit(torch.rand(1, 3, *size, device=device, dtype=dtype))

    @pytest.mark.device_agnostic
    def test_from_config_pretrained(self):
        # vit_s/32 is the checkpoint whose Hub repository does not follow the URL pattern (#5226).
        model = VisionTransformer.from_config("vit_s/32", pretrained=True).eval()
        with torch.no_grad():
            out = model(torch.rand(1, 3, 224, 224))
        assert out.shape == (1, 50, 384)

    @pytest.mark.device_agnostic
    def test_from_config_mode(self, monkeypatch):
        state_dict = VisionTransformer.from_config("vit_ti/16").state_dict()
        monkeypatch.setattr("kornia.models.vit.load_state_dict_from_url", lambda *args, **kwargs: state_dict)

        # ``training`` of the model and of every submodule: ``{False}`` is evaluation mode throughout.
        assert {m.training for m in VisionTransformer.from_config("vit_ti/16", pretrained=True).modules()} == {False}
        assert {m.training for m in VisionTransformer.from_config("vit_ti/16").modules()} == {True}
        assert {m.training for m in VisionTransformer.from_config("vit_ti/16", pretrained=False).modules()} == {True}
