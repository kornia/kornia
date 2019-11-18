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
        inp = torch.rand(2, 1, 32, 32)
        ori = PatchAffineShapeEstimator(32)
        ang = ori(inp)
        assert ang.shape == torch.Size([2, 1, 3])

    def test_print(self):
        sift = PatchAffineShapeEstimator(32)
        sift.__repr__()

    def test_toy(self):
        aff = PatchAffineShapeEstimator(19)
        inp = torch.zeros(1, 1, 19, 19)
        inp[:, :, 5:-5, 1:-1] = 1
        abc = aff(inp)
        expected = torch.tensor([[[0.4146, 0.0000, 1.0000]]])
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
        expected = torch.tensor([[[[36.643, 0., 16.], [0., 10.916, 16.]]]])
        assert_allclose(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 1, 40, 40
        patches = torch.rand(batch_size, channels, height, width)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        laf = torch.tensor([[[[5., 0., 26.], [0., 5., 26.]]]])
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(LAFAffineShapeEstimator(11), (laf, patches),
                         raise_exception=True, rtol=1e-3, atol=1e-3)
