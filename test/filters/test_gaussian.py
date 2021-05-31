import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia
import kornia.testing as utils  # test utils


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


class TestGaussianBlur2d:
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_cardinality(self, batch_shape, device, dtype):
        kernel_size = (5, 7)
        sigma = (1.5, 2.1)

        input = torch.rand(batch_shape, device=device, dtype=dtype)
        actual = kornia.filters.gaussian_blur2d(input, kernel_size, sigma, "replicate")
        assert actual.shape == batch_shape

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        input = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = (3, 3)
        sigma = (1.5, 2.1)
        actual = kornia.filters.gaussian_blur2d(input, kernel_size, sigma, "replicate")
        assert_allclose(actual, actual)

    def test_gradcheck(self, device, dtype):
        # test parameters
        batch_shape = (1, 3, 5, 5)
        kernel_size = (3, 3)
        sigma = (1.5, 2.1)

        # evaluate function gradient
        input = torch.rand(batch_shape, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.gaussian_blur2d, (input, kernel_size, sigma, "replicate"), raise_exception=True)

    def test_jit(self, device, dtype):
        op = kornia.filters.gaussian_blur2d
        op_script = torch.jit.script(op)
        params = [(3, 3), (1.5, 1.5)]

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        assert_allclose(op(img, *params), op_script(img, *params))

    def test_module(self, device, dtype):
        params = [(3, 3), (1.5, 1.5)]
        op = kornia.filters.gaussian_blur2d
        op_module = kornia.filters.GaussianBlur2d(*params)

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        assert_allclose(op(img, *params), op_module(img))
