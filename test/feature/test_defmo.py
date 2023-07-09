import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature import DeFMO
from kornia.testing import assert_close


class TestDeFMO:
    def test_shape(self, device, dtype):
        inp = torch.ones(1, 6, 128, 160, device=device, dtype=dtype)
        defmo = DeFMO().to(device, dtype)
        defmo.eval()  # batchnorm with size 1 is not allowed in train mode
        out = defmo(inp)
        assert out.shape == (1, 24, 4, 128, 160)

    def test_shape_batch(self, device, dtype):
        inp = torch.ones(2, 6, 128, 160, device=device, dtype=dtype)
        defmo = DeFMO().to(device, dtype)
        out = defmo(inp)
        with torch.no_grad():
            assert out.shape == (2, 24, 4, 128, 160)

    def test_gradcheck(self, device, dtype):
        patches = torch.rand(2, 6, 64, 64, device=device, dtype=dtype)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        defmo = DeFMO().to(patches.device, patches.dtype)
        assert gradcheck(defmo, (patches,), eps=1e-4, atol=1e-4, nondet_tol=1e-8, raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 6, 128, 160
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = DeFMO(True).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(DeFMO(True).to(patches.device, patches.dtype).eval())
        with torch.no_grad():
            assert_close(model(patches), model_jit(patches))
