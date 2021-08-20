import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature import DeFMO
from kornia.testing import assert_close


class TestDeFMO:
    @staticmethod
    def test_shape(selfarg, device, dtype):
        inp = torch.ones(1, 6, 240, 320, device=device, dtype=dtype)
        defmo = DeFMO().to(device, dtype)
        defmo.eval()  # batchnorm with size 1 is not allowed in train mode
        out = defmo(inp)
        assert out.shape == (1, 24, 4, 240, 320)

    @staticmethod
    def test_shape_batch(selfarg, device):
        inp = torch.ones(16, 6, 240, 320, device=device)
        defmo = DeFMO().to(device)
        out = defmo(inp)
        assert out.shape == (16, 24, 4, 240, 320)

    @staticmethod
    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(selfarg, device):
        patches = torch.rand(2, 6, 240, 320, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        defmo = DeFMO().to(patches.device, patches.dtype)
        assert gradcheck(defmo, (patches,), eps=1e-4, atol=1e-4, raise_exception=True)

    @staticmethod
    @pytest.mark.jit
    def test_jit(selfarg, device, dtype):
        B, C, H, W = 2, 6, 240, 320
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = DeFMO().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(DeFMO().to(patches.device, patches.dtype).eval())
        assert_close(model(patches), model_jit(patches))
