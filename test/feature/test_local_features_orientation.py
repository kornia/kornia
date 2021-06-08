import pytest
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia
import kornia.testing as utils  # test utils
from kornia.feature.orientation import *


class TestPassLAF:
    @staticmethod
    def test_shape(device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = PassLAF().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    @staticmethod
    def test_shape_batch(device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 34, 2, 3, device=device)
        ori = PassLAF().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    @staticmethod
    def test_print(device):
        sift = PassLAF()
        sift.__repr__()

    @staticmethod
    def test_pass(device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = PassLAF().to(device)
        out = ori(laf, inp)
        assert_allclose(out, laf)

    @staticmethod
    def test_gradcheck(device):
        batch_size, channels, height, width = 1, 1, 21, 21
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        laf = torch.rand(batch_size, 4, 2, 3)
        assert gradcheck(PassLAF().to(device), (patches, laf), raise_exception=True)


class TestPatchDominantGradientOrientation:
    @staticmethod
    def test_shape(device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        ori = PatchDominantGradientOrientation(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([1])

    @staticmethod
    def test_shape_batch(device):
        inp = torch.rand(10, 1, 32, 32, device=device)
        ori = PatchDominantGradientOrientation(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([10])

    @staticmethod
    def test_print(device):
        sift = PatchDominantGradientOrientation(32)
        sift.__repr__()

    @staticmethod
    def test_toy(device):
        ori = PatchDominantGradientOrientation(19).to(device)
        inp = torch.zeros(1, 1, 19, 19, device=device)
        inp[:, :, :10, :] = 1
        ang = ori(inp)
        expected = torch.tensor([90.0], device=device)
        assert_allclose(kornia.rad2deg(ang), expected)

    @staticmethod
    def test_gradcheck(device):
        batch_size, channels, height, width = 1, 1, 13, 13
        ori = PatchDominantGradientOrientation(width).to(device)
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        assert gradcheck(ori, (patches,), raise_exception=True)

    @pytest.mark.jit
    @pytest.mark.skip(" Compiled functions can't take variable number")
    @staticmethod
    def test_jit(device, dtype):
        B, C, H, W = 2, 1, 13, 13
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = PatchDominantGradientOrientation(13).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(PatchDominantGradientOrientation(13).to(patches.device, patches.dtype).eval())
        assert_allclose(model(patches), model_jit(patches))


class TestOriNet:
    @staticmethod
    def test_shape(device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        ori = OriNet().to(device=device, dtype=inp.dtype).eval()
        ang = ori(inp)
        assert ang.shape == torch.Size([1])

    @staticmethod
    def test_pretrained(device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        ori = OriNet(True).to(device=device, dtype=inp.dtype).eval()
        ang = ori(inp)
        assert ang.shape == torch.Size([1])

    @staticmethod
    def test_shape_batch(device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        ori = OriNet(True).to(device=device, dtype=inp.dtype).eval()
        ang = ori(inp)
        assert ang.shape == torch.Size([2])

    @staticmethod
    def test_print(device):
        sift = OriNet(32)
        sift.__repr__()

    @staticmethod
    def test_toy(device):
        inp = torch.zeros(1, 1, 32, 32, device=device)
        inp[:, :, :16, :] = 1
        ori = OriNet(True).to(device=device, dtype=inp.dtype).eval()
        ang = ori(inp)
        expected = torch.tensor([70.58], device=device)
        assert_allclose(kornia.rad2deg(ang), expected, atol=1e-2, rtol=1e-3)

    @pytest.mark.skip("jacobian not well computed")
    @staticmethod
    def test_gradcheck(device):
        batch_size, channels, height, width = 2, 1, 32, 32
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        ori = OriNet().to(device=device, dtype=patches.dtype)
        assert gradcheck(ori, (patches,), raise_exception=True)

    @pytest.mark.jit
    @staticmethod
    def test_jit(device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        tfeat = OriNet(True).to(patches.device, patches.dtype).eval()
        tfeat_jit = torch.jit.script(OriNet(True).to(patches.device, patches.dtype).eval())
        assert_allclose(tfeat_jit(patches), tfeat(patches))


class TestLAFOrienter:
    @staticmethod
    def test_shape(device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFOrienter().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    @staticmethod
    def test_shape_batch(device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 34, 2, 3, device=device)
        ori = LAFOrienter().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    @staticmethod
    def test_print(device):
        sift = LAFOrienter()
        sift.__repr__()

    @staticmethod
    def test_toy(device):
        ori = LAFOrienter(32).to(device)
        inp = torch.zeros(1, 1, 19, 19, device=device)
        inp[:, :, :, :10] = 1
        laf = torch.tensor([[[[0, 5.0, 8.0], [5.0, 0.0, 8.0]]]], device=device)
        new_laf = ori(laf, inp)
        expected = torch.tensor([[[[0.0, 5.0, 8.0], [-5.0, 0, 8.0]]]], device=device)
        assert_allclose(new_laf, expected)

    @staticmethod
    def test_gradcheck(device):
        batch_size, channels, height, width = 1, 1, 21, 21
        patches = torch.rand(batch_size, channels, height, width, device=device).float()
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        laf = torch.ones(batch_size, 2, 2, 3, device=device).float()
        laf[:, :, 0, 1] = 0
        laf[:, :, 1, 0] = 0
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(LAFOrienter(8).to(device), (laf, patches), raise_exception=True, rtol=1e-3, atol=1e-3)
