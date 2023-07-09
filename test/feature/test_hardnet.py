import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature import HardNet, HardNet8
from kornia.testing import assert_close


class TestHardNet:
    def test_shape(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        hardnet = HardNet().to(device)
        hardnet.eval()  # batchnorm with size 1 is not allowed in train mode
        out = hardnet(inp)
        assert out.shape == (1, 128)

    def test_shape_batch(self, device):
        inp = torch.ones(16, 1, 32, 32, device=device)
        hardnet = HardNet().to(device)
        out = hardnet(inp)
        assert out.shape == (16, 128)

    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        hardnet = HardNet().to(patches.device, patches.dtype)
        assert gradcheck(
            hardnet, (patches,), eps=1e-4, atol=1e-4, nondet_tol=1e-8, raise_exception=True, fast_mode=True
        )

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = HardNet().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(HardNet().to(patches.device, patches.dtype).eval())
        assert_close(model(patches), model_jit(patches))


class TestHardNet8:
    def test_shape(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        hardnet = HardNet8().to(device)
        hardnet.eval()  # batchnorm with size 1 is not allowed in train mode
        out = hardnet(inp)
        assert out.shape == (1, 128)

    def test_shape_batch(self, device):
        inp = torch.ones(16, 1, 32, 32, device=device)
        hardnet = HardNet8().to(device)
        out = hardnet(inp)
        assert out.shape == (16, 128)

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        hardnet = HardNet8().to(patches.device, patches.dtype)
        assert gradcheck(hardnet, (patches,), eps=1e-4, atol=1e-4, raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = HardNet8().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(HardNet8().to(patches.device, patches.dtype).eval())
        assert_close(model(patches), model_jit(patches))
