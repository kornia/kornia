import pytest
import torch

import kornia

from testing.base import BaseTester


class TestClassificationHead(BaseTester):
    @pytest.mark.parametrize("B, D, N", [(1, 8, 10), (2, 2, 5)])
    def test_smoke(self, device, dtype, B, D, N):
        feat = torch.rand(B, D, D, device=device, dtype=dtype)
        head = kornia.contrib.ClassificationHead(embed_size=D, num_classes=N).to(device, dtype)
        logits = head(feat)
        assert logits.shape == (B, N)
