import pytest
import kornia.testing as utils  # test utils
import kornia

from torch.testing import assert_allclose
from torch.autograd import gradcheck
from kornia.feature.affine_shape import *


class TestPatchAffineShapeEstimator:
    def test_shape(self, device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        ori = PatchAffineShapeEstimator(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([1, 1, 3])

    def test_shape_batch(self, device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        ori = PatchAffineShapeEstimator(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([2, 1, 3])

    def test_print(self, device):
        sift = PatchAffineShapeEstimator(32)
        sift.__repr__()

    def test_toy(self, device):
        aff = PatchAffineShapeEstimator(19).to(device)
        inp = torch.zeros(1, 1, 19, 19, device=device)
        inp[:, :, 5:-5, 1:-1] = 1
        abc = aff(inp)
        expected = torch.tensor([[[0.4146, 0.0000, 1.0000]]], device=device)
        assert_allclose(abc, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 13, 13
        ori = PatchAffineShapeEstimator(width).to(device)
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        assert gradcheck(ori, (patches, ),
                         raise_exception=True, nondet_tol=1e-4)


class TestLAFAffineShapeEstimator:
    def test_shape(self, device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFAffineShapeEstimator().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_shape_batch(self, device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 34, 2, 3, device=device)
        ori = LAFAffineShapeEstimator().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_print(self, device):
        sift = LAFAffineShapeEstimator()
        sift.__repr__()

    def test_toy(self, device):
        aff = LAFAffineShapeEstimator(32).to(device)
        inp = torch.zeros(1, 1, 32, 32, device=device)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[20., 0., 16.], [0., 20., 16.]]]], device=device)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[36.643, 0., 16.], [0., 10.916, 16.]]]], device=device)
        assert_allclose(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 40, 40
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        laf = torch.tensor([[[[5., 0., 26.], [0., 5., 26.]]]], device=device)
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(LAFAffineShapeEstimator(11).to(device), (laf, patches),
                         raise_exception=True, rtol=1e-3, atol=1e-3, nondet_tol=1e-4)
