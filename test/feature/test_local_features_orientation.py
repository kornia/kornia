import pytest
import kornia.testing as utils  # test utils
import kornia

from torch.testing import assert_allclose
from torch.autograd import gradcheck
from kornia.feature.orientation import *


class TestPassLAF:
    def test_shape(self):
        inp = torch.rand(1, 1, 32, 32)
        laf = torch.rand(1, 1, 2, 3)
        ori = PassLAF()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_shape_batch(self):
        inp = torch.rand(2, 1, 32, 32)
        laf = torch.rand(2, 34, 2, 3)
        ori = PassLAF()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_print(self):
        sift = PassLAF()
        sift.__repr__()

    def test_pass(self):
        inp = torch.rand(1, 1, 32, 32)
        laf = torch.rand(1, 1, 2, 3)
        ori = PassLAF()
        out = ori(laf, inp)
        assert_allclose(out, laf)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 1, 41, 41
        patches = torch.rand(batch_size, channels, height, width)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        laf = torch.rand(batch_size, 4, 2, 3)
        assert gradcheck(PassLAF(), (patches, laf),
                         raise_exception=True)


class TestPatchDominantGradientOrientation:
    def test_shape(self):
        inp = torch.rand(1, 1, 32, 32)
        ori = PatchDominantGradientOrientation(32)
        ang = ori(inp)
        assert ang.shape == torch.Size([1])

    def test_shape_batch(self):
        inp = torch.rand(10, 1, 32, 32)
        ori = PatchDominantGradientOrientation(32)
        ang = ori(inp)
        assert ang.shape == torch.Size([10])

    def test_print(self):
        sift = PatchDominantGradientOrientation(32)
        sift.__repr__()

    def test_toy(self):
        ori = PatchDominantGradientOrientation(19)
        inp = torch.zeros(1, 1, 19, 19)
        inp[:, :, :10, :] = 1
        ang = ori(inp)
        expected = torch.tensor([90.])
        assert_allclose(kornia.rad2deg(ang), expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 1, 13, 13
        ori = PatchDominantGradientOrientation(width)
        patches = torch.rand(batch_size, channels, height, width)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        assert gradcheck(ori, (patches, ),
                         raise_exception=True)


class TestLAFOrienter:
    def test_shape(self):
        inp = torch.rand(1, 1, 32, 32)
        laf = torch.rand(1, 1, 2, 3)
        ori = LAFOrienter()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_shape_batch(self):
        inp = torch.rand(2, 1, 32, 32)
        laf = torch.rand(2, 34, 2, 3)
        ori = LAFOrienter()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_print(self):
        sift = LAFOrienter()
        sift.__repr__()

    def test_toy(self):
        ori = LAFOrienter(32)
        inp = torch.zeros(1, 1, 19, 19)
        inp[:, :, :10, :] = 1
        laf = torch.tensor([[[[0, 5., 8.], [5.0, 0., 8.]]]])
        new_laf = ori(laf, inp)
        expected = torch.tensor([[[[5., 0., 8.], [0., 5., 8.]]]])
        assert_allclose(new_laf, expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 1, 21, 21
        patches = torch.rand(batch_size, channels, height, width).float()
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        laf = torch.ones(batch_size, 2, 2, 3).float()
        laf[:, :, 0, 1] = 0
        laf[:, :, 1, 0] = 0
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(LAFOrienter(8), (laf, patches),
                         raise_exception=True, rtol=1e-3, atol=1e-3)
