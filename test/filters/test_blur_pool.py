import pytest
import torch

from kornia.filters import (
    BlurPool2D,
    EdgeAwareBlurPool2D,
    MaxBlurPool2D,
    blur_pool2d,
    edge_aware_blur_pool2d,
    max_blur_pool2d,
)
from kornia.testing import BaseTester, tensor_to_gradcheck_var


class TestMaxBlurPool(BaseTester):
    @pytest.mark.parametrize('kernel_size', [3, (5, 5)])
    @pytest.mark.parametrize('ceil_mode', [True, False])
    def test_smoke(self, kernel_size, ceil_mode, device, dtype):
        inpt = torch.rand(1, 1, 10, 10, device=device, dtype=dtype)
        actual = MaxBlurPool2D(kernel_size, ceil_mode=ceil_mode)(inpt)

        assert actual.shape == (1, 1, 5, 5)

    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize('kernel_size', [3, (5, 5)])
    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_cardinality(self, batch_size, kernel_size, ceil_mode, device, dtype):
        inpt = torch.zeros(batch_size, 4, 4, 8, device=device, dtype=dtype)
        blur = MaxBlurPool2D(kernel_size, ceil_mode=ceil_mode)
        assert blur(inpt).shape == (batch_size, 4, 2, 4)

    def test_exception(self):
        inpt = torch.rand(1, 1, 3, 3)
        with pytest.raises(Exception) as errinfo:
            MaxBlurPool2D((3, 5))(inpt)
        assert 'Invalid kernel shape. Expect CxC_outxNxN' in str(errinfo)

    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_noncontiguous(self, batch_size, device, dtype):
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = max_blur_pool2d(inp, 3)

        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = tensor_to_gradcheck_var(img)  # to var
        self.gradcheck(max_blur_pool2d, (img, 3))

    @pytest.mark.parametrize('kernel_size', [(3, 3), 5])
    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_module(self, kernel_size, batch_size, device, dtype):
        op = max_blur_pool2d
        op_module = MaxBlurPool2D

        img = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        actual = op_module(kernel_size)(img)
        expected = op(img, kernel_size)
        self.assert_close(actual, expected)

    @pytest.mark.slow
    @pytest.mark.parametrize('kernel_size', [3, (5, 5)])
    @pytest.mark.parametrize('batch_size', [1, 2])
    @pytest.mark.parametrize('ceil_mode', [True, False])
    def test_dynamo(self, batch_size, kernel_size, ceil_mode, device, dtype, torch_optimizer):
        inpt = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = MaxBlurPool2D(kernel_size, ceil_mode=ceil_mode)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt), op_optimized(inpt))


class TestBlurPool(BaseTester):
    @pytest.mark.parametrize('kernel_size', [3, (5, 5)])
    @pytest.mark.parametrize('stride', [1, 2])
    def test_smoke(self, kernel_size, stride, device, dtype):
        inpt = torch.rand(1, 1, 10, 10, device=device, dtype=dtype)
        actual = BlurPool2D(kernel_size, stride=stride)(inpt)
        expected = (1, 1, int(10 / stride), int(10 / stride))
        assert actual.shape == expected

    @pytest.mark.parametrize('kernel_size', [3, (5, 5)])
    @pytest.mark.parametrize('batch_size', [1, 2])
    @pytest.mark.parametrize('stride', [1, 2])
    def test_cardinality(self, batch_size, kernel_size, stride, device, dtype):
        inpt = torch.zeros(batch_size, 4, 4, 8, device=device, dtype=dtype)
        actual = BlurPool2D(kernel_size, stride=stride)(inpt)
        expected = (batch_size, 4, int(4 / stride), int(8 / stride))
        assert actual.shape == expected

    def test_exception(self):
        inpt = torch.rand(1, 1, 3, 3)
        with pytest.raises(Exception) as errinfo:
            BlurPool2D((3, 5))(inpt)
        assert 'Invalid kernel shape. Expect CxC_(out, None)xNxN' in str(errinfo)

    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_noncontiguous(self, batch_size, device, dtype):
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = blur_pool2d(inp, 3)
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = tensor_to_gradcheck_var(img)  # to var
        self.gradcheck(blur_pool2d, (img, 3))

    @pytest.mark.parametrize('kernel_size', [3, (5, 5)])
    @pytest.mark.parametrize('batch_size', [1, 2])
    @pytest.mark.parametrize('stride', [1, 2])
    def test_module(self, batch_size, kernel_size, stride, device, dtype):
        op = blur_pool2d
        op_module = BlurPool2D

        img = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        actual = op_module(kernel_size)(img)
        expected = op(img, kernel_size)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize('kernel_size', [3, (5, 5)])
    @pytest.mark.parametrize('batch_size', [1, 2])
    @pytest.mark.parametrize('stride', [1, 2])
    def test_dynamo(self, batch_size, kernel_size, stride, device, dtype, torch_optimizer):
        inpt = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = BlurPool2D(kernel_size, stride=stride)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt), op_optimized(inpt))


class TestEdgeAwareBlurPool(BaseTester):
    @pytest.mark.parametrize('kernel_size', [3, (5, 5)])
    @pytest.mark.parametrize('batch_size', [1, 2])
    @pytest.mark.parametrize('edge_threshold', [1.25, 2.5])
    @pytest.mark.parametrize('edge_dilation_kernel_size', [3, 5])
    def test_smoke(self, kernel_size, batch_size, edge_threshold, edge_dilation_kernel_size, device, dtype):
        inpt = torch.zeros(batch_size, 3, 8, 8, device=device, dtype=dtype)
        actual = edge_aware_blur_pool2d(inpt, kernel_size, edge_threshold, edge_dilation_kernel_size)
        assert actual.shape == inpt.shape

    @pytest.mark.parametrize('kernel_size', [3, (5, 5)])
    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_cardinality(self, kernel_size, batch_size, device, dtype):
        inp = torch.zeros(batch_size, 3, 8, 8, device=device, dtype=dtype)
        blur = edge_aware_blur_pool2d(inp, kernel_size=kernel_size)
        assert blur.shape == inp.shape

    def test_exception(self):
        with pytest.raises(Exception) as errinfo:
            inpt = torch.rand(1, 3, 3)
            edge_aware_blur_pool2d(inpt, 3)
        assert "shape must be [[\'B\', \'C\', \'H\', \'W\']]" in str(errinfo)
        with pytest.raises(Exception) as errinfo:
            inpt = torch.rand(1, 1, 3, 3)
            edge_aware_blur_pool2d(inpt, 3, edge_threshold=-1)
        assert "edge threshold should be positive, but got" in str(errinfo)

    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_noncontiguous(self, batch_size, device, dtype):
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = edge_aware_blur_pool2d(inp, 3)
        assert actual.is_contiguous()

    @pytest.mark.parametrize('kernel_size', [3, (5, 5)])
    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_module(self, kernel_size, batch_size, device, dtype):
        op = edge_aware_blur_pool2d
        op_module = EdgeAwareBlurPool2D

        img = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        actual = op_module(kernel_size)(img)
        expected = op(img, kernel_size)
        self.assert_close(actual, expected)

    def test_gradcheck(self, device):
        img = torch.rand((1, 2, 5, 4), device=device)
        img = tensor_to_gradcheck_var(img)  # to var
        self.gradcheck(edge_aware_blur_pool2d, (img, 3))

    def test_smooth(self, device, dtype):
        img = torch.ones(1, 1, 5, 5, device=device, dtype=dtype)
        img[0, 0, :, :2] = 0
        blur = edge_aware_blur_pool2d(img, kernel_size=3, edge_threshold=32.0)
        self.assert_close(img, blur)

    @pytest.mark.slow
    @pytest.mark.parametrize('kernel_size', [3, (5, 5)])
    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_dynamo(self, batch_size, kernel_size, device, dtype, torch_optimizer):
        op = edge_aware_blur_pool2d
        inpt = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        op = EdgeAwareBlurPool2D(kernel_size)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt), op_optimized(inpt))
