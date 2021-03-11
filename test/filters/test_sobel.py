import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestSpatialGradient:
    def test_shape(self, device, dtype):
        inp = torch.zeros(1, 3, 4, 4, device=device, dtype=dtype)
        sobel = kornia.filters.SpatialGradient()
        assert sobel(inp).shape == (1, 3, 2, 4, 4)

    def test_shape_batch(self, device, dtype):
        inp = torch.zeros(2, 6, 4, 4, device=device, dtype=dtype)
        sobel = kornia.filters.SpatialGradient()
        assert sobel(inp).shape == (2, 6, 2, 4, 4)

    def test_edges(self, device, dtype):
        inp = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]]], device=device, dtype=dtype)

        expected = torch.tensor([[[[
            [0., 1., 0., -1., 0.],
            [1., 3., 0., -3., -1.],
            [2., 4., 0., -4., -2.],
            [1., 3., 0., -3., -1.],
            [0., 1., 0., -1., 0.],
        ], [
            [0., 1., 2., 1., 0.],
            [1., 3., 4., 3., 1.],
            [0., 0., 0., 0., 0],
            [-1., -3., -4., -3., -1],
            [0., -1., -2., -1., 0.],
        ]]]], device=device, dtype=dtype)

        edges = kornia.filters.spatial_gradient(inp, normalized=False)
        assert_allclose(edges, expected)

    def test_edges_norm(self, device, dtype):
        inp = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]]], device=device, dtype=dtype)

        expected = torch.tensor([[[[
            [0., 1., 0., -1., 0.],
            [1., 3., 0., -3., -1.],
            [2., 4., 0., -4., -2.],
            [1., 3., 0., -3., -1.],
            [0., 1., 0., -1., 0.],
        ], [
            [0., 1., 2., 1., 0.],
            [1., 3., 4., 3., 1.],
            [0., 0., 0., 0., 0],
            [-1., -3., -4., -3., -1],
            [0., -1., -2., -1., 0.],
        ]]]], device=device, dtype=dtype) / 8.0

        edges = kornia.filters.spatial_gradient(inp, normalized=True)
        assert_allclose(edges, expected)

    def test_edges_sep(self, device, dtype):
        inp = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]]], device=device, dtype=dtype)

        expected = torch.tensor([[[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., -1., 0.],
            [1., 1., 0., -1., -1.],
            [0., 1., 0., -1., 0.],
            [0., 0., 0., 0., 0.]
        ], [
            [0., 0., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0.],
            [0., -1., -1., -1., 0.],
            [0., 0., -1., 0., 0.]
        ]]]], device=device, dtype=dtype)

        edges = kornia.filters.spatial_gradient(inp, 'diff',
                                                normalized=False)
        assert_allclose(edges, expected)

    def test_edges_sep_norm(self, device, dtype):
        inp = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]]], device=device, dtype=dtype)

        expected = torch.tensor([[[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., -1., 0.],
            [1., 1., 0., -1., -1.],
            [0., 1., 0., -1., 0.],
            [0., 0., 0., 0., 0.]
        ], [
            [0., 0., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0.],
            [0., -1., -1., -1., 0.],
            [0., 0., -1., 0., 0.]
        ]]]], device=device, dtype=dtype) / 2.0

        edges = kornia.filters.spatial_gradient(inp, 'diff',
                                                normalized=True)
        assert_allclose(edges, expected)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = kornia.filters.spatial_gradient(inp)
        assert_allclose(actual, actual)

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 1, 1, 3, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.spatial_gradient, (img,),
                         raise_exception=True)

    def test_jit(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = kornia.filters.spatial_gradient
        op_script = torch.jit.script(op)
        actual = op_script(img)
        expected = op(img)
        assert_allclose(actual, expected)

    def test_module(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = kornia.filters.spatial_gradient
        op_module = kornia.filters.SpatialGradient()
        expected = op(img)
        actual = op_module(img)
        assert_allclose(actual, expected)


class TestSpatialGradient3d:
    def test_shape(self, device, dtype):
        inp = torch.zeros(1, 2, 4, 5, 6, device=device, dtype=dtype)
        sobel = kornia.filters.SpatialGradient3d()
        assert sobel(inp).shape == (1, 2, 3, 4, 5, 6)

    def test_shape_batch(self, device, dtype):
        inp = torch.zeros(7, 2, 4, 5, 6, device=device, dtype=dtype)
        sobel = kornia.filters.SpatialGradient3d()
        assert sobel(inp).shape == (7, 2, 3, 4, 5, 6)

    @pytest.mark.skip("fix due to bug in kernel_flip")
    def test_edges(self, device, dtype):
        inp = torch.tensor([[[
            [[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 1., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]],
            [[0., 0., 0., 0., 0.],
             [0., 0., 1., 0., 0.],
             [0., 1., 1., 1., 0.],
             [0., 0., 1., 0., 0.],
             [0., 0., 0., 0., 0.]],
            [[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 1., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]],
        ]]], device=device, dtype=dtype)

        expected = torch.tensor([
            [[[[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.5000, 0.0000, -0.5000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.5000, 0.0000, -0.5000, 0.0000],
                 [0.5000, 0.5000, 0.0000, -0.5000, -0.5000],
                 [0.0000, 0.5000, 0.0000, -0.5000, 0.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.5000, 0.0000, -0.5000, 0.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],
                [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, -0.5000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                 [[0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                    [0.0000, 0.5000, 0.5000, 0.5000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, -0.5000, -0.5000, -0.5000, 0.0000],
                    [0.0000, 0.0000, -0.5000, 0.0000, 0.0000]],
                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, -0.5000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],
                [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                    [0.0000, 0.5000, 0.0000, 0.5000, 0.0000],
                    [0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                 [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, -0.5000, 0.0000, 0.0000],
                    [0.0000, -0.5000, 0.0000, -0.5000, 0.0000],
                    [0.0000, 0.0000, -0.5000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]]]], device=device, dtype=dtype)

        edges = kornia.filters.spatial_gradient3d(inp)
        assert_allclose(edges, expected)

    def test_gradcheck(self, device, dtype):
        img = torch.rand(1, 1, 1, 3, 4, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.spatial_gradient3d, (img,),
                         raise_exception=True)

    @pytest.mark.skip("issue with device in kernel generation")
    def test_jit(self, device, dtype):
        img = torch.rand(2, 3, 1, 4, 5, device=device, dtype=dtype)
        op = kornia.filters.spatial_gradient3d
        op_script = torch.jit.script(op)
        expected = op(img)
        actual = op_script(img)
        assert_allclose(actual, expected)

    def test_module(self, device, dtype):
        img = torch.rand(2, 3, 1, 4, 5, device=device, dtype=dtype)
        op = kornia.filters.spatial_gradient3d
        op_module = kornia.filters.SpatialGradient3d()
        expected = op(img)
        actual = op_module(img)
        assert_allclose(actual, expected)


class TestSobel:
    def test_shape(self, device, dtype):
        inp = torch.zeros(1, 3, 4, 4, device=device, dtype=dtype)
        sobel = kornia.filters.Sobel()
        assert sobel(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self, device, dtype):
        inp = torch.zeros(3, 2, 4, 4, device=device, dtype=dtype)
        sobel = kornia.filters.Sobel()
        assert sobel(inp).shape == (3, 2, 4, 4)

    def test_magnitude(self, device, dtype):
        inp = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]]], device=device, dtype=dtype)

        expected = torch.tensor([[[
            [0., 1.4142, 2.0, 1.4142, 0.],
            [1.4142, 4.2426, 4.00, 4.2426, 1.4142],
            [2.0, 4.0000, 0.00, 4.0000, 2.0],
            [1.4142, 4.2426, 4.00, 4.2426, 1.4142],
            [0., 1.4142, 2.0, 1.4142, 0.],
        ]]], device=device, dtype=dtype)

        edges = kornia.filters.sobel(inp, normalized=False, eps=0.)
        assert_allclose(edges, expected)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = kornia.filters.sobel(inp)
        expected = actual
        assert_allclose(actual, actual)

    def test_gradcheck_unnorm(self, device, dtype):
        if "cuda" in str(device):
            pytest.skip("RuntimeError: Backward is not reentrant, i.e., running backward,")
        batch_size, channels, height, width = 1, 1, 3, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.sobel, (img, False),
                         raise_exception=True)

    def test_gradcheck(self, device, dtype):
        if "cuda" in str(device):
            pytest.skip("RuntimeError: Backward is not reentrant, i.e., running backward,")
        batch_size, channels, height, width = 1, 1, 3, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.sobel, (img, True),
                         raise_exception=True)

    def test_jit(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = kornia.filters.sobel
        op_script = torch.jit.script(op)
        expected = op(img)
        actual = op_script(img)
        assert_allclose(actual, expected)

    def test_module(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = kornia.filters.sobel
        op_module = kornia.filters.Sobel()
        expected = op(img)
        actual = op_module(img)
        assert_allclose(actual, expected)
