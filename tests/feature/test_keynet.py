import torch

from kornia.feature import KeyNet
from testing.base import BaseTester


class TestKeyNet(BaseTester):
    def test_shape(self, device, dtype):
        inp = torch.rand(1, 1, 16, 16, device=device, dtype=dtype)
        keynet = KeyNet().to(device, dtype)
        out = keynet(inp)
        assert out.shape == inp.shape

    def test_shape_batch(self, device, dtype):
        inp = torch.ones(16, 1, 16, 16, device=device, dtype=dtype)
        keynet = KeyNet().to(device, dtype)
        out = keynet(inp)
        assert out.shape == inp.shape

    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 16, 16, device=device, dtype=torch.float64)
        keynet = KeyNet().to(patches.device, patches.dtype)
        self.gradcheck(keynet, (patches,), eps=1e-4, atol=1e-4, nondet_tol=1e-8)
