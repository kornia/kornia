import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils
from kornia.feature.sold2 import SOLD2, SOLD2_detector
from kornia.testing import assert_close


class TestSOLD2_detector:
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_shape(self, device, batch_size):
        inp = torch.ones(batch_size, 1, 128, 128, device=device)
        sold2 = SOLD2_detector(pretrained=False).to(device)
        sold2.eval()
        out = sold2(inp)
        assert out["junction_heatmap"].shape == (batch_size, 128, 128)
        assert out["line_heatmap"].shape == (batch_size, 128, 128)

    @pytest.mark.skip("Takes ages to run")
    def test_gradcheck(self, device):
        img = torch.rand(2, 1, 128, 128, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        sold2 = SOLD2_detector(pretrained=False).to(img.device, img.dtype)

        def proxy_forward(x):
            return sold2.forward(x)["junction_heatmap"]

        assert gradcheck(proxy_forward, (img,), eps=1e-4, atol=1e-4, raise_exception=True)

    @pytest.mark.skip("Does not like recursive definition of Hourglass in backbones.py l.134.")
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 128, 128
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = SOLD2_detector().to(img.device, img.dtype).eval()
        model_jit = torch.jit.script(model)
        assert_close(model(img), model_jit(img))


class TestSOLD2:
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_shape(self, device, batch_size):
        inp = torch.ones(batch_size, 1, 256, 256, device=device)
        sold2 = SOLD2(pretrained=False).to(device)
        sold2.eval()
        out = sold2(inp)
        assert out["dense_desc"].shape == (batch_size, 128, 64, 64)

    @pytest.mark.skip("Takes ages to run")
    def test_gradcheck(self, device):
        img = torch.rand(2, 1, 256, 256, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        sold2 = SOLD2(pretrained=False).to(img.device, img.dtype)

        def proxy_forward(x):
            return sold2.forward(x)["dense_desc"]

        assert gradcheck(proxy_forward, (img,), eps=1e-4, atol=1e-4, raise_exception=True)

    @pytest.mark.skip("Does not like recursive definition of Hourglass in backbones.py l.134.")
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 256, 256
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = SOLD2().to(img.device, img.dtype).eval()
        model_jit = torch.jit.script(model)
        assert_close(model(img), model_jit(img))
