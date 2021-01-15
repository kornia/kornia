import pytest

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck

from kornia.feature import TFeat
import kornia.testing as utils  # test utils


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

    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        tfeat = TFeat().to(patches.device, patches.dtype)
        assert gradcheck(tfeat, (patches,), eps=1e-2, atol=1e-2,
                         raise_exception=True, )
