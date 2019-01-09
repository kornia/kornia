import pytest

import torch
import torchgeometry.image as image
from torch.autograd import gradcheck

import utils
from common import TEST_DEVICES

@pytest.mark.parametrize("window_size", [5, 11, 15])
@pytest.mark.parametrize("sigma", [1.5, 5.0, 21.0])
def test_get_gaussian_kernel(window_size, sigma):
    kernel = image.get_gaussian_kernel(window_size, sigma)
    assert kernel.shape == (window_size,)
    assert kernel.sum().item() == pytest.approx(1.0)

@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize("sigma", [1.5, 21.0])
def test_get_gaussian_kernel2d(ksize_x, ksize_y, sigma):
    kernel = image.get_gaussian_kernel2d(
        (ksize_x, ksize_y), (sigma, sigma))
    assert kernel.shape == (ksize_x, ksize_y)
    assert kernel.sum().item() == pytest.approx(1.0)

@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize("sigma", [1.5, 21.0])
@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_shape",
    [(1, 1, 10, 16), (1, 4, 8, 15), (2, 3, 11, 7)])
def test_gaussian_blur(batch_shape, device_type, ksize_x, ksize_y, sigma):
    kernel_size = (ksize_x, ksize_y)
    sigma = (sigma, sigma)

    input = torch.rand(batch_shape).to(torch.device(device_type))
    gauss = image.GaussianBlur(kernel_size, sigma)
    assert gauss(input).shape == batch_shape

    # functional
    assert image.gaussian_blur(input, kernel_size, sigma).shape == batch_shape

    # evaluate function gradient
    input = utils.tensor_to_gradcheck_var(input)  # to var
    assert gradcheck(gauss, (input,), raise_exception=True)
