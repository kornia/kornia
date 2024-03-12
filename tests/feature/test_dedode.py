import sys

import pytest
import torch

from kornia.feature.dedode import DeDoDe
from kornia.utils._compat import torch_version_le


class TestDeDoDe:
    @pytest.mark.skipif(torch_version_le(2, 0, 0), reason="Autocast not supported")
    def test_smoke(self, dtype, device):
        # only testing "B" as dinov2 is quite heavy
        dedode = DeDoDe(descriptor_model="B").to(device, dtype)
        shape = (2, 3, 128, 128)
        n = 1000
        inp = torch.randn(*shape, device=device, dtype=dtype)
        keypoints, scores, descriptions = dedode(inp, n=n)
        assert keypoints.shape == (shape[0], n, 2)
        assert scores.shape == (shape[0], n)
        assert descriptions.shape == (shape[0], n, 256)

    @pytest.mark.slow
    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    def load_weights(self, dtype, device):
        # only testing "B" as dinov2 is quite heavy
        dedode = DeDoDe(descriptor_model="B").to(device, dtype)
        dedode = DeDoDe(descriptor_model="G").to(device, dtype)
        print(dedode)
