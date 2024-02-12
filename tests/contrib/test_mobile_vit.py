import pytest
import torch

import kornia

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
        mvit = kornia.contrib.MobileViT(mode=mode, patch_size=patch_size).to(device, dtype)

        out = mvit(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (B, channel[mode], 8, 8)
