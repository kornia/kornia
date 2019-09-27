import pytest
import kornia.testing as utils  # test utils
import kornia

from torch.testing import assert_allclose
from torch.autograd import gradcheck
from kornia.feature.affine_shape import *


class TestPatchAffineShapeEstimator:
    def test_shape(self):
        inp = torch.rand(1, 1, 32, 32)
        ori = PatchAffineShapeEstimator(32)
        ang = ori(inp)
        assert ang.shape == torch.Size([1, 1, 3])

    def test_shape_batch(self):
        inp = torch.rand(10, 1, 32, 32)
        ori = PatchAffineShapeEstimator(32)
        ang = ori(inp)
        assert ang.shape == torch.Size([10, 1, 3])

    def test_print(self):
        sift = PatchAffineShapeEstimator(32)
        sift.__repr__()

    def test_toy(self):
        aff = PatchAffineShapeEstimator(19)
        inp = torch.zeros(1, 1, 19, 19)
        inp[:, :, 5:-5, 1:-1] = 1
        abc = aff(inp)
        expected = torch.tensor([[[0.0229, 0.0000, 0.0450]]])
        assert_allclose(abc, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 1, 13, 13
        ori = PatchAffineShapeEstimator(width)
        patches = torch.rand(batch_size, channels, height, width)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        assert gradcheck(ori, (patches, ),
                         raise_exception=True)


class TestLAFAffineShapeEstimator:
    def test_shape(self):
        inp = torch.rand(1, 1, 32, 32)
        laf = torch.rand(1, 1, 2, 3)
        ori = LAFAffineShapeEstimator()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_shape_batch(self):
        inp = torch.rand(2, 1, 32, 32)
        laf = torch.rand(2, 34, 2, 3)
        ori = LAFAffineShapeEstimator()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_print(self):
        sift = LAFAffineShapeEstimator()
        sift.__repr__()

    def test_toy(self):
        aff = LAFAffineShapeEstimator(32)
        inp = torch.zeros(1, 1, 32, 32)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[20., 0., 16.], [0., 20., 16.]]]])
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[36.246, 0., 16.], [0., 11.036, 16.]]]])
        assert_allclose(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 1, 21, 21
        patches = torch.rand(batch_size, channels, height, width).float()
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        laf = torch.ones(batch_size, 4, 2, 3).float()
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(LAFAffineShapeEstimator(8), (laf, patches),
                         raise_exception=True, rtol=1e-4, atol=1e-4)
