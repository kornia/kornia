import pytest
import torch

from kornia.feature import SOSNet
from testing.base import BaseTester


class TestSOSNet(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        sosnet = SOSNet(pretrained=False).to(device)
        sosnet.eval()  # batchnorm with size 1 is not allowed in train mode
        out = sosnet(inp)
        assert out.shape == (1, 128)

    def test_shape_batch(self, device):
        inp = torch.ones(16, 1, 32, 32, device=device)
        sosnet = SOSNet(pretrained=False).to(device)
        out = sosnet(inp)
        assert out.shape == (16, 128)

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=torch.float64)
        sosnet = SOSNet(pretrained=False).to(patches.device, patches.dtype)
        self.gradcheck(sosnet, (patches,), eps=1e-4, atol=1e-4)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = SOSNet().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(SOSNet().to(patches.device, patches.dtype).eval())
        self.assert_close(model(patches), model_jit(patches))
