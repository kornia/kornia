import pytest
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia.testing as utils  # test utils
from kornia.feature.affine_shape import *


class TestPatchAffineShapeEstimator:
    @staticmethod
    def test_shape(device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        ori = PatchAffineShapeEstimator(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([1, 1, 3])

    @staticmethod
    def test_shape_batch(device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        ori = PatchAffineShapeEstimator(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([2, 1, 3])

    @staticmethod
    def test_print(device):
        sift = PatchAffineShapeEstimator(32)
        sift.__repr__()

    @staticmethod
    def test_toy(device):
        aff = PatchAffineShapeEstimator(19).to(device)
        inp = torch.zeros(1, 1, 19, 19, device=device)
        inp[:, :, 5:-5, 1:-1] = 1
        abc = aff(inp)
        expected = torch.tensor([[[0.4146, 0.0000, 1.0000]]], device=device)
        assert_allclose(abc, expected, atol=1e-4, rtol=1e-4)

    @staticmethod
    def test_gradcheck(device):
        batch_size, channels, height, width = 1, 1, 13, 13
        ori = PatchAffineShapeEstimator(width).to(device)
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        assert gradcheck(ori, (patches,), raise_exception=True, nondet_tol=1e-4)

    @pytest.mark.jit
    @staticmethod
    def test_jit(device, dtype):
        B, C, H, W = 2, 1, 13, 13
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        tfeat = PatchAffineShapeEstimator(W).to(patches.device, patches.dtype).eval()
        tfeat_jit = torch.jit.script(PatchAffineShapeEstimator(W).to(patches.device, patches.dtype).eval())
        assert_allclose(tfeat_jit(patches), tfeat(patches))


class TestLAFAffineShapeEstimator:
    @staticmethod
    def test_shape(device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFAffineShapeEstimator().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    @staticmethod
    def test_shape_batch(device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 34, 2, 3, device=device)
        ori = LAFAffineShapeEstimator().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    @staticmethod
    def test_print(device):
        sift = LAFAffineShapeEstimator()
        sift.__repr__()

    @staticmethod
    def test_toy(device):
        aff = LAFAffineShapeEstimator(32).to(device)
        inp = torch.zeros(1, 1, 32, 32, device=device)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[20.0, 0.0, 16.0], [0.0, 20.0, 16.0]]]], device=device)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[36.643, 0.0, 16.0], [0.0, 10.916, 16.0]]]], device=device)
        assert_allclose(new_laf, expected, atol=1e-4, rtol=1e-4)

    @staticmethod
    def test_gradcheck(device):
        batch_size, channels, height, width = 1, 1, 40, 40
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        laf = torch.tensor([[[[5.0, 0.0, 26.0], [0.0, 5.0, 26.0]]]], device=device)
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(
            LAFAffineShapeEstimator(11).to(device),
            (laf, patches),
            raise_exception=True,
            rtol=1e-3,
            atol=1e-3,
            nondet_tol=1e-4,
        )

    @pytest.mark.jit
    @pytest.mark.skip("Failing because of extract patches")
    @staticmethod
    def test_jit(device, dtype):
        B, C, H, W = 1, 1, 13, 13
        inp = torch.zeros(B, C, H, W, device=device)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[20.0, 0.0, 16.0], [0.0, 20.0, 16.0]]]], device=device)
        tfeat = LAFAffineShapeEstimator(W).to(inp.device, inp.dtype).eval()
        tfeat_jit = torch.jit.script(LAFAffineShapeEstimator(W).to(inp.device, inp.dtype).eval())
        assert_allclose(tfeat_jit(laf, inp), tfeat(laf, inp))


class TestLAFAffNetShapeEstimator:
    @staticmethod
    def test_shape(device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFAffNetShapeEstimator(False).to(device).eval()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    @staticmethod
    def test_pretrained(device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFAffNetShapeEstimator(True).to(device).eval()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    @staticmethod
    def test_shape_batch(device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 5, 2, 3, device=device)
        ori = LAFAffNetShapeEstimator().to(device).eval()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    @staticmethod
    def test_print(device):
        sift = LAFAffNetShapeEstimator()
        sift.__repr__()

    @staticmethod
    def test_toy(device):
        aff = LAFAffNetShapeEstimator(True).to(device).eval()
        inp = torch.zeros(1, 1, 32, 32, device=device)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[20.0, 0.0, 16.0], [0.0, 20.0, 16.0]]]], device=device)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[40.8758, 0.0, 16.0], [-0.3824, 9.7857, 16.0]]]], device=device)
        assert_allclose(new_laf, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.skip("jacobian not well computed")
    @staticmethod
    def test_gradcheck(device):
        batch_size, channels, height, width = 1, 1, 35, 35
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        laf = torch.tensor([[[[8.0, 0.0, 16.0], [0.0, 8.0, 16.0]]]], device=device)
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(
            LAFAffNetShapeEstimator(True).to(device, dtype=patches.dtype),
            (laf, patches),
            raise_exception=True,
            rtol=1e-3,
            atol=1e-3,
            nondet_tol=1e-4,
        )

    @pytest.mark.jit
    @pytest.mark.skip("Laf type is not a torch.Tensor????")
    @staticmethod
    def test_jit(device, dtype):
        B, C, H, W = 1, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        laf = torch.tensor([[[[8.0, 0.0, 16.0], [0.0, 8.0, 16.0]]]], device=device)
        laf_estimator = LAFAffNetShapeEstimator(True).to(device, dtype=patches.dtype).eval()
        laf_estimator_jit = torch.jit.script(LAFAffNetShapeEstimator(True).to(device, dtype=patches.dtype).eval())
        assert_allclose(laf_estimator(laf, patches), laf_estimator_jit(laf, patches))
