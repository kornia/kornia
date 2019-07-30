import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestSpatialGradient:
    def test_shape(self):
        inp = torch.zeros(1, 3, 4, 4)
        sobel = kornia.filters.SpatialGradient()
        assert sobel(inp).shape == (1, 3, 2, 4, 4)

    def test_shape_batch(self):
        inp = torch.zeros(2, 6, 4, 4)
        sobel = kornia.filters.SpatialGradient()
        assert sobel(inp).shape == (2, 6, 2, 4, 4)

    def test_edges(self):
        inp = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]]])

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
        ]]]])

        edges = kornia.filters.spatial_gradient(inp)
        assert_allclose(edges, expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.spatial_gradient, (img,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return kornia.filters.spatial_gradient(input)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img)
        expected = kornia.filters.spatial_gradient(img)
        assert_allclose(actual, expected)


class TestSobel:
    def test_shape(self):
        inp = torch.zeros(1, 3, 4, 4)
        sobel = kornia.filters.Sobel()
        assert sobel(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self):
        inp = torch.zeros(3, 2, 4, 4)
        sobel = kornia.filters.Sobel()
        assert sobel(inp).shape == (3, 2, 4, 4)

    def test_magnitude(self):
        inp = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]]])

        expected = torch.tensor([[[
            [0., 1.4142, 2.0, 1.4142, 0.],
            [1.4142, 4.2426, 4.00, 4.2426, 1.4142],
            [2.0, 4.0000, 0.00, 4.0000, 2.0],
            [1.4142, 4.2426, 4.00, 4.2426, 1.4142],
            [0., 1.4142, 2.0, 1.4142, 0.],
        ]]])

        edges = kornia.filters.sobel(inp)
        assert_allclose(edges, expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.sobel, (img,), raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return kornia.filters.sobel(input)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img)
        expected = kornia.filters.sobel(img)
        assert_allclose(actual, expected)
