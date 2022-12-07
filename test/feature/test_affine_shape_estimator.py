import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature.affine_shape import LAFAffineShapeEstimator, LAFAffNetShapeEstimator, PatchAffineShapeEstimator
from kornia.testing import assert_close


# TODO: add kornia.testing.BaseTester
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
        assert_close(abc, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 13, 13
        ori = PatchAffineShapeEstimator(width).to(device)
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        assert gradcheck(ori, (patches,), raise_exception=True, nondet_tol=1e-4, fast_mode=True)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 13, 13
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        tfeat = PatchAffineShapeEstimator(W).to(patches.device, patches.dtype).eval()
        tfeat_jit = torch.jit.script(PatchAffineShapeEstimator(W).to(patches.device, patches.dtype).eval())
        assert_close(tfeat_jit(patches), tfeat(patches))


# TODO: add kornia.testing.BaseTester
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

    def test_toy(self, device, dtype):
        aff = LAFAffineShapeEstimator(32, preserve_orientation=False).to(device, dtype)
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[20.0, 0.0, 16.0], [0.0, 20.0, 16.0]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[36.643, 0.0, 16.0], [0.0, 10.916, 16.0]]]], device=device, dtype=dtype)
        assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_toy_preserve(self, device, dtype):
        aff = LAFAffineShapeEstimator(32, preserve_orientation=True).to(device, dtype)
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[0.0, 20.0, 16.0], [-20.0, 0.0, 16.0]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[0.0, 36.643, 16.0], [-10.916, 0, 16.0]]]], device=device, dtype=dtype)
        assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_toy_not_preserve(self, device):
        aff = LAFAffineShapeEstimator(32, preserve_orientation=False).to(device)
        inp = torch.zeros(1, 1, 32, 32, device=device)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[0.0, 20.0, 16.0], [-20.0, 0.0, 16.0]]]], device=device)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[36.643, 0, 16.0], [0, 10.916, 16.0]]]], device=device)
        assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
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
            fast_mode=True,
        )

    @pytest.mark.jit
    @pytest.mark.skip("Failing because of extract patches")
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 1, 13, 13
        inp = torch.zeros(B, C, H, W, device=device)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[20.0, 0.0, 16.0], [0.0, 20.0, 16.0]]]], device=device)
        tfeat = LAFAffineShapeEstimator(W).to(inp.device, inp.dtype).eval()
        tfeat_jit = torch.jit.script(LAFAffineShapeEstimator(W).to(inp.device, inp.dtype).eval())
        assert_close(tfeat_jit(laf, inp), tfeat(laf, inp))


# TODO: add kornia.testing.BaseTester
class TestLAFAffNetShapeEstimator:
    def test_shape(self, device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFAffNetShapeEstimator(False).to(device).eval()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_pretrained(self, device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFAffNetShapeEstimator(True).to(device).eval()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_shape_batch(self, device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 5, 2, 3, device=device)
        ori = LAFAffNetShapeEstimator().to(device).eval()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_print(self, device):
        sift = LAFAffNetShapeEstimator()
        sift.__repr__()

    def test_toy(self, device, dtype):
        aff = LAFAffNetShapeEstimator(True).to(device, dtype).eval()
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[20.0, 0.0, 16.0], [0.0, 20.0, 16.0]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[40.8758, 0.0, 16.0], [-0.3824, 9.7857, 16.0]]]], device=device, dtype=dtype)
        assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
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
            fast_mode=True,
        )

    @pytest.mark.jit
    @pytest.mark.skip("Laf type is not a torch.Tensor????")
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        laf = torch.tensor([[[[8.0, 0.0, 16.0], [0.0, 8.0, 16.0]]]], device=device)
        laf_estimator = LAFAffNetShapeEstimator(True).to(device, dtype=patches.dtype).eval()
        laf_estimator_jit = torch.jit.script(LAFAffNetShapeEstimator(True).to(device, dtype=patches.dtype).eval())
        assert_close(laf_estimator(laf, patches), laf_estimator_jit(laf, patches))
