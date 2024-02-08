import pytest
import torch

from kornia.feature.siftdesc import DenseSIFTDescriptor
from kornia.feature.siftdesc import SIFTDescriptor
from kornia.feature.siftdesc import get_sift_bin_ksize_stride_pad
from kornia.feature.siftdesc import get_sift_pooling_kernel

from testing.base import BaseTester


@pytest.mark.parametrize("ksize", [5, 13, 25])
def test_get_sift_pooling_kernel(ksize):
    kernel = get_sift_pooling_kernel(ksize)
    assert kernel.shape == (ksize, ksize)


@pytest.mark.parametrize("ps,n_bins,ksize,stride,pad", [(41, 3, 20, 13, 5), (32, 4, 12, 8, 3)])
def test_get_sift_bin_ksize_stride_pad(ps, n_bins, ksize, stride, pad):
    out = get_sift_bin_ksize_stride_pad(ps, n_bins)
    assert out == (ksize, stride, pad)


class TestSIFTDescriptor(BaseTester):
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
        assert out.shape == (3, (3**2) * 5)

    def test_toy(self, device, dtype):
        patch = torch.ones(1, 1, 6, 6, device=device, dtype=dtype)
        patch[0, 0, :, 3:] = 0
        sift = SIFTDescriptor(6, num_ang_bins=4, num_spatial_bins=1, clipval=0.2, rootsift=False).to(device, dtype)
        out = sift(patch)
        expected = torch.tensor([[0, 0, 1.0, 0]], device=device, dtype=dtype)
        self.assert_close(out, expected, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        dtype = torch.float64
        batch_size, channels, height, width = 1, 1, 15, 15
        patches = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        sift = SIFTDescriptor(15).to(device, dtype)
        self.gradcheck(sift, (patches,), nondet_tol=1e-4)

    @pytest.mark.skip("Compiled functions can't take variable number")
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = SIFTDescriptor(41).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(SIFTDescriptor(41).to(patches.device, patches.dtype).eval())
        self.assert_close(model(patches), model_jit(patches))


class TestDenseSIFTDescriptor(BaseTester):
    def test_shape_default(self, device, dtype):
        bs, h, w = 1, 20, 15
        inp = torch.rand(1, 1, h, w, device=device, dtype=dtype)
        sift = DenseSIFTDescriptor().to(device, dtype)
        out = sift(inp)
        assert out.shape == torch.Size([bs, 128, h, w])

    def test_batch_shape(self, device, dtype):
        bs, h, w = 2, 32, 15
        inp = torch.rand(bs, 1, h, w, device=device, dtype=dtype)
        sift = DenseSIFTDescriptor().to(device, dtype)
        out = sift(inp)
        assert out.shape == torch.Size([bs, 128, h, w])

    def test_batch_shape_custom(self, device, dtype):
        bs, h, w = 2, 40, 30
        inp = torch.rand(bs, 1, h, w, device=device, dtype=dtype)
        sift = DenseSIFTDescriptor(5, 3, 3, padding=1, stride=2).to(device, dtype)
        out = sift(inp)
        assert out.shape == torch.Size([bs, 45, h // 2, w // 2])

    def test_print(self, device):
        sift = DenseSIFTDescriptor()
        sift.__repr__()

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 16, 16
        patches = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(DenseSIFTDescriptor(4, 2, 2), (patches), nondet_tol=1e-4)
