from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature import KeyNet
from kornia.testing import assert_close


class TestKeyNet:
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

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 16, 16, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        keynet = KeyNet().to(patches.device, patches.dtype)
        assert gradcheck(keynet, (patches,), eps=1e-4, atol=1e-4, raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = KeyNet(True).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(KeyNet(True).to(patches.device, patches.dtype).eval())
        assert_close(model(patches), model_jit(patches))
