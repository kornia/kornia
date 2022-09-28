from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestMaxBlurPool:
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_shape(self, ceil_mode, device, dtype):
        inp = torch.zeros(1, 4, 4, 8, device=device, dtype=dtype)
        blur = kornia.filters.MaxBlurPool2D(3, ceil_mode=ceil_mode)
        assert blur(inp).shape == (1, 4, 2, 4)

    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_shape_batch(self, ceil_mode, device, dtype):
        inp = torch.zeros(2, 4, 4, 8, device=device, dtype=dtype)
        blur = kornia.filters.MaxBlurPool2D(3, ceil_mode=ceil_mode)
        assert blur(inp).shape == (2, 4, 2, 4)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = 3
        actual = kornia.filters.max_blur_pool2d(inp, kernel_size)
        assert_close(actual, actual)

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.max_blur_pool2d, (img, 3), raise_exception=True)

    def test_jit(self, device, dtype):
        op = kornia.filters.max_blur_pool2d
        op_script = torch.jit.script(op)

        kernel_size = 3
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        actual = op_script(img, kernel_size)
        expected = op(img, kernel_size)
        assert_close(actual, expected)

    def test_module(self, device, dtype):
        op = kornia.filters.max_blur_pool2d
        op_module = kornia.filters.MaxBlurPool2D

        kernel_size = 3
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        actual = op_module(kernel_size)(img)
        expected = op(img, kernel_size)
        assert_close(actual, expected)


class TestBlurPool:
    def test_shape(self, device, dtype):
        inp = torch.zeros(1, 4, 4, 8, device=device, dtype=dtype)
        blur = kornia.filters.BlurPool2D(3, stride=1)
        assert blur(inp).shape == (1, 4, 4, 8)
        blur = kornia.filters.BlurPool2D(3)
        assert blur(inp).shape == (1, 4, 2, 4)

    def test_shape_batch(self, device, dtype):
        inp = torch.zeros(2, 4, 4, 8, device=device, dtype=dtype)
        blur = kornia.filters.BlurPool2D(3, stride=1)
        assert blur(inp).shape == (2, 4, 4, 8)
        blur = kornia.filters.BlurPool2D(3)
        assert blur(inp).shape == (2, 4, 2, 4)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = 3
        actual = kornia.filters.blur_pool2d(inp, kernel_size)
        assert_close(actual, actual)

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.blur_pool2d, (img, 3), raise_exception=True)

    def test_jit(self, device, dtype):
        op = kornia.filters.blur_pool2d
        op_script = torch.jit.script(op)

        kernel_size = 3
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        actual = op_script(img, kernel_size)
        expected = op(img, kernel_size)
        assert_close(actual, expected)

    def test_module(self, device, dtype):
        op = kornia.filters.blur_pool2d
        op_module = kornia.filters.BlurPool2D

        kernel_size = 3
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        actual = op_module(kernel_size)(img)
        expected = op(img, kernel_size)
        assert_close(actual, expected)


class TestEdgeAwareBlurPool:
    def test_shape(self, device, dtype):
        inp = torch.zeros(1, 3, 8, 8, device=device, dtype=dtype)
        blur = kornia.filters.edge_aware_blur_pool2d(inp, kernel_size=3)
        assert blur.shape == inp.shape

    def test_shape_batch(self, device, dtype):
        inp = torch.zeros(2, 3, 8, 8, device=device, dtype=dtype)
        blur = kornia.filters.edge_aware_blur_pool2d(inp, kernel_size=3)
        assert blur.shape == inp.shape

    def test_gradcheck(self, device, dtype):
        img = torch.rand((1, 2, 5, 4), device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.edge_aware_blur_pool2d, (img, 3), raise_exception=True)

    def test_jit(self, device, dtype):
        op = kornia.filters.edge_aware_blur_pool2d
        op_script = torch.jit.script(op)

        kernel_size = 3
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        actual = op_script(img, kernel_size)
        expected = op(img, kernel_size)
        assert_close(actual, expected)

    def test_smooth(self, device, dtype):
        img = torch.ones(1, 1, 5, 5).to(device, dtype)
        img[0, 0, :, :2] = 0
        blur = kornia.filters.edge_aware_blur_pool2d(img, kernel_size=3, edge_threshold=32.0)
        assert_close(img, blur)
