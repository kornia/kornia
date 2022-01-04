import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature.siftdesc import SIFTDescriptor, get_sift_bin_ksize_stride_pad, get_sift_pooling_kernel
from kornia.testing import assert_close


@pytest.mark.parametrize("ksize", [5, 13, 25])
def test_get_sift_pooling_kernel(ksize):
    kernel = get_sift_pooling_kernel(ksize)
    assert kernel.shape == (ksize, ksize)


@pytest.mark.parametrize("ps,n_bins,ksize,stride,pad", [(41, 3, 20, 13, 5), (32, 4, 12, 8, 3)])
def test_get_sift_bin_ksize_stride_pad(ps, n_bins, ksize, stride, pad):
    out = get_sift_bin_ksize_stride_pad(ps, n_bins)
    assert out == (ksize, stride, pad)


class TestSIFTDescriptor:
    def test_shape(self, device, dtype):
        inp = torch.ones(1, 1, 32, 32, device=device, dtype=dtype)
        sift = SIFTDescriptor(32).to(device, dtype)
        out = sift(inp)
        assert out.shape == (1, 128)

    def test_batch_shape(self, device, dtype):
        inp = torch.ones(2, 1, 15, 15, device=device, dtype=dtype)
        sift = SIFTDescriptor(15).to(device, dtype)
        out = sift(inp)
        assert out.shape == (2, 128)

    def test_batch_shape_non_std(self, device, dtype):
        inp = torch.ones(3, 1, 19, 19, device=device, dtype=dtype)
        sift = SIFTDescriptor(19, 5, 3).to(device, dtype)
        out = sift(inp)
        assert out.shape == (3, (3 ** 2) * 5)

    def test_toy(self, device, dtype):
        patch = torch.ones(1, 1, 6, 6, device=device, dtype=dtype)
        patch[0, 0, :, 3:] = 0
        sift = SIFTDescriptor(6, num_ang_bins=4, num_spatial_bins=1, clipval=0.2, rootsift=False).to(device, dtype)
        out = sift(patch)
        expected = torch.tensor([[0, 0, 1.0, 0]], device=device, dtype=dtype)
        assert_close(out, expected, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        dtype = torch.float64
        batch_size, channels, height, width = 1, 1, 15, 15
        patches = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        sift = SIFTDescriptor(15).to(device, dtype)
        assert gradcheck(sift, (patches,), raise_exception=True, nondet_tol=1e-4)

    @pytest.mark.skip("Compiled functions can't take variable number")
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = SIFTDescriptor(32).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(model)
        assert_close(model(patches), model_jit(patches))
