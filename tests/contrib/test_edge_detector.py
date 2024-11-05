import pytest
import torch

import kornia

from testing.base import BaseTester


class TestEdgeDetector(BaseTester):
    @pytest.mark.slow
    def test_smoke(self, device, dtype):
        img = torch.rand(2, 3, 64, 64, device=device, dtype=dtype)
        net = kornia.contrib.EdgeDetector().to(device, dtype)
        out = net(img)
        assert out.shape == (2, 1, 64, 64)

    @pytest.mark.slow
    @pytest.mark.skip(reason="issue with `ClassVar[list[int]]`")
    def test_jit(self, device, dtype):
        op = kornia.contrib.EdgeDetector().to(device, dtype)
        op_jit = torch.jit.script(op)
        assert op_jit is not None
