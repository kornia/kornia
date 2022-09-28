from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature import HyNet
from kornia.testing import assert_close


class TestHyNet:
    def test_shape(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        hynet = HyNet().to(device)
        out = hynet(inp)
        assert out.shape == (1, 128)

    def test_shape_batch(self, device):
        inp = torch.ones(16, 1, 32, 32, device=device)
        hynet = HyNet().to(device)
        out = hynet(inp)
        assert out.shape == (16, 128)

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        hynet = HyNet().to(patches.device, patches.dtype)
        assert gradcheck(hynet, (patches,), eps=1e-4, atol=1e-4, raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = HyNet().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(model)
        assert_close(model(patches), model_jit(patches))
