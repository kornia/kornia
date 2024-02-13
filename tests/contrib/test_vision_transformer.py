import pytest
import torch

import kornia

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
        vit = kornia.contrib.VisionTransformer(image_size=image_size, num_heads=H, embed_dim=D).to(device, dtype)

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
            kornia.contrib.VisionTransformer(image_size=image_size, num_heads=H, embed_dim=D).to(device, dtype)

    def test_backbone(self, device, dtype):
        def backbone_mock(x):
            return torch.ones(1, 128, 14, 14, device=device, dtype=dtype)

        img = torch.rand(1, 3, 32, 32, device=device, dtype=dtype)
        vit = kornia.contrib.VisionTransformer(backbone=backbone_mock, num_heads=8).to(device, dtype)
        out = vit(img)
        assert out.shape == (1, 197, 128)
