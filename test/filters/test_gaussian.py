import pytest

import kornia
import kornia.testing as utils  # test utils

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [1.5, 5.0])
def test_get_gaussian_kernel(window_size, sigma):
    kernel = kornia.get_gaussian_kernel1d(window_size, sigma)
    assert kernel.shape == (window_size,)
    assert kernel.sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [1.5, 5.0])
def test_get_gaussian_discrete_kernel(window_size, sigma):
    kernel = kornia.get_gaussian_discrete_kernel1d(window_size, sigma)
    assert kernel.shape == (window_size,)
    assert kernel.sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [1.5, 5.0])
def test_get_gaussian_erf_kernel(window_size, sigma):
    kernel = kornia.get_gaussian_erf_kernel1d(window_size, sigma)
    assert kernel.shape == (window_size,)
    assert kernel.sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize("sigma", [1.5, 2.1])
def test_get_gaussian_kernel2d(ksize_x, ksize_y, sigma):
    kernel = kornia.get_gaussian_kernel2d((ksize_x, ksize_y), (sigma, sigma))
    assert kernel.shape == (ksize_x, ksize_y)
    assert kernel.sum().item() == pytest.approx(1.0)


class TestGaussianBlur:
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_gaussian_blur(self, batch_shape, device):
        kernel_size = (5, 7)
        sigma = (1.5, 2.1)

        input = torch.rand(batch_shape).to(device)
        gauss = kornia.filters.GaussianBlur2d(kernel_size, sigma, "replicate")
        assert gauss(input).shape == batch_shape

    def test_noncontiguous(self, device):
        batch_size = 3
        inp = torch.rand(3, 5, 5).expand(batch_size, -1, -1, -1).to(device)

        kernel_size = (3, 3)
        sigma = (1.5, 2.1)
        gauss = kornia.filters.GaussianBlur2d(kernel_size, sigma, "replicate")
        actual = gauss(inp)
        expected = actual
        assert_allclose(actual, actual)

    def test_gradcheck(self, device):
        # test parameters
        batch_shape = (2, 3, 11, 7)
        kernel_size = (5, 3)
        sigma = (1.5, 2.1)

        # evaluate function gradient
        input = torch.rand(batch_shape).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(
            kornia.gaussian_blur2d,
            (input, kernel_size, sigma, "replicate"),
            raise_exception=True,
        )

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(img):

            return kornia.gaussian_blur2d(img, (5, 5), (1.2, 1.2), "replicate")

        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width).to(device)
        expected = kornia.filters.GaussianBlur2d(
            (5, 5), (1.2, 1.2), "replicate"
        )(img)
        actual = op_script(img)
        assert_allclose(actual, expected)


@pytest.mark.parametrize("window_size", [5])
def test_get_laplacian_kernel(window_size):
    kernel = kornia.get_laplacian_kernel1d(window_size)
    assert kernel.shape == (window_size,)
    assert kernel.sum().item() == pytest.approx(0.0)


@pytest.mark.parametrize("window_size", [7])
def test_get_laplacian_kernel2d(window_size):
    kernel = kornia.get_laplacian_kernel2d(window_size)
    assert kernel.shape == (window_size, window_size)
    assert kernel.sum().item() == pytest.approx(0.0)
    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, -48.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert_allclose(expected, kernel)


class TestLaplacian:
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_laplacian(self, batch_shape, device):
        kernel_size = 5

        input = torch.rand(batch_shape).to(device)
        laplace = kornia.filters.Laplacian(kernel_size)
        assert laplace(input).shape == batch_shape

    def test_noncontiguous(self, device):
        batch_size = 3
        inp = torch.rand(3, 5, 5).expand(batch_size, -1, -1, -1).to(device)

        kernel_size = 3
        laplace = kornia.filters.Laplacian(kernel_size)
        actual = laplace(inp)
        expected = actual
        assert_allclose(actual, actual)

    def test_gradcheck(self, device):
        # test parameters
        batch_shape = (2, 3, 11, 7)
        kernel_size = 9

        # evaluate function gradient
        input = torch.rand(batch_shape).to(device)
        input = utils.tensor_to_gradcheck_var(input)
        assert gradcheck(
            kornia.laplacian, (input, kernel_size), raise_exception=True)
