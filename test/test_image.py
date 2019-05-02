import pytest

import torch
import torchgeometry.image as image
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import utils
from common import device_type

@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [1.5, 5.0])
def test_get_gaussian_kernel(window_size, sigma):
    kernel = image.get_gaussian_kernel(window_size, sigma)
    assert kernel.shape == (window_size,)
    assert kernel.sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize("sigma", [1.5, 2.1])
def test_get_gaussian_kernel2d(ksize_x, ksize_y, sigma):
    kernel = image.get_gaussian_kernel2d(
        (ksize_x, ksize_y), (sigma, sigma))
    assert kernel.shape == (ksize_x, ksize_y)
    assert kernel.sum().item() == pytest.approx(1.0)


class TestGaussianBlur:
    @pytest.mark.parametrize("batch_shape",
                             [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_gaussian_blur(self, batch_shape, device_type):
        kernel_size = (5, 7)
        sigma = (1.5, 2.1)

        input = torch.rand(batch_shape).to(torch.device(device_type))
        gauss = image.GaussianBlur(kernel_size, sigma)
        assert gauss(input).shape == batch_shape

    def test_gradcheck(self):
        # test parameters
        batch_shape = (2, 3, 11, 7)
        kernel_size = (5, 3)
        sigma = (1.5, 2.1)

        # evaluate function gradient
        input = torch.rand(batch_shape)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(image.gaussian_blur, (input, kernel_size, sigma,),
                         raise_exception=True)

class TestNormalize:
    def test_smoke(self):
        mean = [0.5]
        std = [0.1]
        repr = 'Normalize(mean=[0.5], std=[0.1])'
        assert str(image.Normalize(mean, std)) == repr

    def test_normalize(self):

        # prepare input data
        data = torch.ones(1, 2, 2)
        mean = torch.tensor([0.5])
        std = torch.tensor([2.0])

        # expected output
        expected = torch.tensor([0.25]).repeat(1, 2, 2).view_as(data)

        f = image.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_batch_normalize(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2
        mean = torch.tensor([0.5, 1.0, 2.0])
        std = torch.tensor([2., 2., 2.])

        # expected output
        expected = torch.tensor([1.25, 1., 0.5]).repeat(2, 1, 1).view_as(data)

        f = image.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_gradcheck(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2
        mean = torch.tensor([0.5, 1.0, 2.0]).double()
        std = torch.tensor([2., 2., 2.]).double()

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(image.Normalize(mean, std), (data,),
                         raise_exception=True)
