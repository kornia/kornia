import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature import TFeat
from kornia.testing import assert_close


class TestTFeat:
    def test_shape(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        tfeat = TFeat().to(device)
        tfeat.eval()  # batchnorm with size 1 is not allowed in train mode
        out = tfeat(inp)
        assert out.shape == (1, 128)

    def test_pretrained(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        tfeat = TFeat(True).to(device)
        tfeat.eval()  # batchnorm with size 1 is not allowed in train mode
        out = tfeat(inp)
        assert out.shape == (1, 128)

    def test_shape_batch(self, device):
        inp = torch.ones(16, 1, 32, 32, device=device)
        tfeat = TFeat().to(device)
        out = tfeat(inp)
        assert out.shape == (16, 128)

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        tfeat = TFeat().to(patches.device, patches.dtype)
        assert gradcheck(tfeat, (patches,), eps=1e-2, atol=1e-2, raise_exception=True, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        tfeat = TFeat(True).to(patches.device, patches.dtype).eval()
        tfeat_jit = torch.jit.script(TFeat(True).to(patches.device, patches.dtype).eval())
        assert_close(tfeat_jit(patches), tfeat(patches))
