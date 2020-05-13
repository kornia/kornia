import pytest
import kornia.testing as utils  # test utils
import kornia

from torch.testing import assert_allclose
from torch.autograd import gradcheck
from kornia.feature.orientation import *


class TestPassLAF:
    def test_shape(self, device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = PassLAF().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_shape_batch(self, device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 34, 2, 3, device=device)
        ori = PassLAF().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_print(self, device):
        sift = PassLAF()
        sift.__repr__()

    def test_pass(self, device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = PassLAF().to(device)
        out = ori(laf, inp)
        assert_allclose(out, laf)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 21, 21
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        laf = torch.rand(batch_size, 4, 2, 3)
        assert gradcheck(PassLAF().to(device), (patches, laf),
                         raise_exception=True)


class TestPatchDominantGradientOrientation:
    def test_shape(self, device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        ori = PatchDominantGradientOrientation(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([1])

    def test_shape_batch(self, device):
        inp = torch.rand(10, 1, 32, 32, device=device)
        ori = PatchDominantGradientOrientation(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([10])

    def test_print(self, device):
        sift = PatchDominantGradientOrientation(32)
        sift.__repr__()

    def test_toy(self, device):
        ori = PatchDominantGradientOrientation(19).to(device)
        inp = torch.zeros(1, 1, 19, 19, device=device)
        inp[:, :, :10, :] = 1
        ang = ori(inp)
        expected = torch.tensor([90.], device=device)
        assert_allclose(kornia.rad2deg(ang), expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 13, 13
        ori = PatchDominantGradientOrientation(width).to(device)
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        assert gradcheck(ori, (patches, ),
                         raise_exception=True)


class TestLAFOrienter:
    def test_shape(self, device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFOrienter().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_shape_batch(self, device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 34, 2, 3, device=device)
        ori = LAFOrienter().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_print(self, device):
        sift = LAFOrienter()
        sift.__repr__()

    def test_toy(self, device):
        ori = LAFOrienter(32).to(device)
        inp = torch.zeros(1, 1, 19, 19, device=device)
        inp[:, :, :10, :] = 1
        laf = torch.tensor([[[[0, 5., 8.], [5.0, 0., 8.]]]], device=device)
        new_laf = ori(laf, inp)
        expected = torch.tensor([[[[5., 0., 8.], [0., 5., 8.]]]], device=device)
        assert_allclose(new_laf, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 21, 21
        patches = torch.rand(batch_size, channels, height, width, device=device).float()
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        laf = torch.ones(batch_size, 2, 2, 3, device=device).float()
        laf[:, :, 0, 1] = 0
        laf[:, :, 1, 0] = 0
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(LAFOrienter(8).to(device), (laf, patches),
                         raise_exception=True, rtol=1e-3, atol=1e-3)
