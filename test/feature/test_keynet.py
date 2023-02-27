import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature import KeyNet


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

    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 16, 16, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        keynet = KeyNet().to(patches.device, patches.dtype)
        assert gradcheck(keynet, (patches,), eps=1e-4, atol=1e-4, nondet_tol=1e-8, raise_exception=True, fast_mode=True)
