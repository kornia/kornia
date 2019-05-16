import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose
from common import device_type

import torchgeometry.image as image
import utils


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

    def test_jit(self):
        @torch.jit.script
        def op_script(img):

            return image.gaussian_blur(img, (5, 5), (1.2, 1.2))

        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width)
        expected = image.GaussianBlur((5, 5), (1.2, 1.2))(img)
        actual = op_script(img)
        assert_allclose(actual, expected)


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

    def test_broadcast_normalize(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0])
        std = torch.tensor([2.0, 2.0, 2.0])

        # expected output
        expected = torch.tensor([1.25, 1, 0.5]).repeat(2, 1, 1).view_as(data)

        f = image.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_batch_normalize(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0]).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0]).repeat(2, 1)

        # expected output
        expected = torch.tensor([1.25, 1, 0.5]).repeat(2, 1, 1).view_as(data)

        f = image.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor, mean: torch.Tensor,
                      std: torch.Tensor) -> torch.Tensor:

            return image.normalize(data, mean, std)

            data = torch.ones(2, 3, 1, 1)
            data += 2

            mean = torch.tensor([0.5, 1.0, 2.0]).repeat(2, 1)
            std = torch.tensor([2.0, 2.0, 2.0]).repeat(2, 1)

            actual = op_script(data, mean, std)
            expected = image.normalize(data, mean, std)
            assert_allclose(actual, expected)

    def test_gradcheck(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2
        mean = torch.tensor([0.5, 1.0, 2.0]).double()
        std = torch.tensor([2., 2., 2.]).double()

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(image.Normalize(mean, std), (data,),
                         raise_exception=True)


@pytest.mark.parametrize("window_size", [5])
def test_get_laplacian_kernel(window_size):
    kernel = image.get_laplacian_kernel(window_size)
    assert kernel.shape == (window_size,)
    assert kernel.sum().item() == pytest.approx(0.0)


@pytest.mark.parametrize("window_size", [7])
def test_get_laplacian_kernel2d(window_size):
    kernel = image.get_laplacian_kernel2d(window_size)
    assert kernel.shape == (window_size, window_size)
    assert kernel.sum().item() == pytest.approx(0.0)
    expected = torch.tensor([[1., 1., 1., 1., 1., 1., 1.],
                             [1., 1., 1., 1., 1., 1., 1.],
                             [1., 1., 1., 1., 1., 1., 1.],
                             [1., 1., 1., -48., 1., 1., 1.],
                             [1., 1., 1., 1., 1., 1., 1.],
                             [1., 1., 1., 1., 1., 1., 1.],
                             [1., 1., 1., 1., 1., 1., 1.]])
    assert_allclose(expected, kernel)


class TestLaplacian:
    @pytest.mark.parametrize("batch_shape",
                             [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_laplacian(self, batch_shape, device_type):
        kernel_size = 5

        input = torch.rand(batch_shape).to(torch.device(device_type))
        laplace = image.Laplacian(kernel_size)
        assert laplace(input).shape == batch_shape

    def test_gradcheck(self):
        # test parameters
        batch_shape = (2, 3, 11, 7)
        kernel_size = 9

        # evaluate function gradient
        input = torch.rand(batch_shape)
        input = utils.tensor_to_gradcheck_var(input)
        assert gradcheck(image.laplacian, (input, kernel_size,),
                         raise_exception=True)


class TestHsvToRgb:

    def test_hsv_to_rgb(self):

        expected = torch.tensor([[[21., 22.],
                                  [22., 22.]],

                                 [[13., 14.],
                                  [14., 14.]],

                                 [[8., 8.],
                                  [8., 8.]]])

        data = torch.tensor([[[0.0641, 0.0714],
                              [0.0714, 0.0714]],

                             [[0.6190, 0.6364],
                              [0.6364, 0.6364]],

                             [[21.0000, 22.0000],
                              [22.0000, 22.0000]]])

        f = image.HsvToRgb()
        assert_allclose(f(data), expected, atol=1e-3, rtol=1e-3)

    def test_batch_hsv_to_rgb(self):

        expected = torch.tensor([[[21., 22.],
                                  [22., 22.]],

                                 [[13., 14.],
                                  [14., 14.]],

                                 [[8., 8.],
                                  [8., 8.]]])  # 3x2x2

        data = torch.tensor([[[0.0641, 0.0714],
                              [0.0714, 0.0714]],

                             [[0.6190, 0.6364],
                              [0.6364, 0.6364]],

                             [[21.0000, 22.0000],
                              [22.0000, 22.0000]]])  # 3x2x2

        f = image.HsvToRgb()
        data = data.repeat(2, 1, 1, 1)  # 2x3x2x2
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x2x2
        assert_allclose(f(data), expected, atol=1e-3, rtol=1e-3)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return image.hsv_to_rgb(data)
            data = torch.tensor([[[[21., 22.],
                                   [22., 22.]],

                                  [[13., 14.],
                                   [14., 14.]],

                                  [[8., 8.],
                                   [8., 8.]]]])  # 3x2x2

            actual = op_script(data)
            expected = image.hsv_to_rgb(data)
            assert_allclose(actual, expected)


class TestRgbToHsv:

    def test_rgb_to_hsv(self):

        data = torch.tensor([[[21., 22.],
                              [22., 22.]],

                             [[13., 14.],
                              [14., 14.]],

                             [[8., 8.],
                              [8., 8.]]])

        expected = torch.tensor([[[0.0641, 0.0714],
                                  [0.0714, 0.0714]],

                                 [[0.6190, 0.6364],
                                  [0.6364, 0.6364]],

                                 [[21.0000/255, 22.0000/255],
                                  [22.0000/255, 22.0000/255]]])

        f = image.RgbToHsv()
        assert_allclose(f(data/255), expected, atol=1e-4, rtol=1e-5)

    def test_batch_rgb_to_hsv(self):

        data = torch.tensor([[[21., 22.],
                              [22., 22.]],

                             [[13., 14.],
                              [14., 14.]],

                             [[8., 8.],
                              [8., 8.]]])  # 3x2x2

        expected = torch.tensor([[[0.0641, 0.0714],
                                  [0.0714, 0.0714]],

                                 [[0.6190, 0.6364],
                                  [0.6364, 0.6364]],

                                 [[21.0000/255, 22.0000/255],
                                  [22.0000/255, 22.0000/255]]])  # 3x2x2
        f = image.RgbToHsv()
        data = data.repeat(2, 1, 1, 1)  # 2x3x2x2
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x2x2
        assert_allclose(f(data/255), expected, atol=1e-4, rtol=1e-5)

    def test_gradcheck(self):

        data = torch.tensor([[[[21., 22.],
                               [22., 22.]],

                              [[13., 14.],
                               [14., 14.]],

                              [[8., 8.],
                               [8., 8.]]]])  # 3x2x2

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(image.RgbToHsv(), (data,),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return image.rgb_to_hsv(data)
            data = torch.tensor([[[[21., 22.],
                                   [22., 22.]],

                                  [[13., 14.],
                                   [14., 14.]],

                                  [[8., 8.],
                                   [8., 8.]]]])  # 3x2x2

            actual = op_script(data)
            expected = image.rgb_to_hsv(data)
            assert_allclose(actual, expected)


class TestBgrToRgb:

    def test_bgr_to_rgb(self):

        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[2., 2.],
                              [2., 2.]],

                             [[3., 3.],
                              [3., 3.]]])  # 3x2x2

        expected = torch.tensor([[[3., 3.],
                                  [3., 3.]],

                                 [[2., 2.],
                                  [2., 2.]],

                                 [[1., 1.],
                                  [1., 1.]]])  # 3x2x2

        f = image.BgrToRgb()
        assert_allclose(f(data), expected)

    def test_batch_bgr_to_rgb(self):

        # prepare input data
        data = torch.tensor([[[[1., 1.],
                               [1., 1.]],

                              [[2., 2.],
                               [2., 2.]],

                              [[3., 3.],
                               [3., 3.]]],

                             [[[1., 1.],
                               [1., 1.]],

                              [[2., 2.],
                               [2., 2.]],

                              [[3., 3.],
                               [3., 3.]]]])  # 2x3x2x2

        expected = torch.tensor([[[[3., 3.],
                                   [3., 3.]],

                                  [[2., 2.],
                                   [2., 2.]],

                                  [[1., 1.],
                                   [1., 1.]]],

                                 [[[3., 3.],
                                   [3., 3.]],

                                  [[2., 2.],
                                   [2., 2.]],

                                  [[1., 1.],
                                   [1., 1.]]]])  # 2x3x2x2

        f = image.BgrToRgb()
        out = f(data)
        assert_allclose(out, expected)

    def test_gradcheck(self):

        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[2., 2.],
                              [2., 2.]],

                             [[3., 3.],
                              [3., 3.]]])  # 3x2x2

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(image.BgrToRgb(), (data,),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return image.bgr_to_rgb(data)

            data = torch.tensor([[[1., 1.],
                                  [1., 1.]],

                                 [[2., 2.],
                                  [2., 2.]],

                                 [[3., 3.],
                                  [3., 3.]]])  # 3x2x2

            actual = op_script(data)
            expected = image.bgr_to_rgb(data)
            assert_allclose(actual, expected)


class TestRgbToBgr:

    def test_rgb_to_bgr(self):

        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[2., 2.],
                              [2., 2.]],

                             [[3., 3.],
                              [3., 3.]]])  # 3x2x2

        expected = torch.tensor([[[3., 3.],
                                  [3., 3.]],

                                 [[2., 2.],
                                  [2., 2.]],

                                 [[1., 1.],
                                  [1., 1.]]])  # 3x2x2

        f = image.RgbToBgr()
        assert_allclose(f(data), expected)

    def test_gradcheck(self):

        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[2., 2.],
                              [2., 2.]],

                             [[3., 3.],
                              [3., 3.]]])  # 3x2x2

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(image.RgbToBgr(), (data,),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return image.rgb_to_bgr(data)

            data = torch.tensor([[[1., 1.],
                                  [1., 1.]],

                                 [[2., 2.],
                                  [2., 2.]],

                                 [[3., 3.],
                                  [3., 3.]]])  # 3x2x2

            actual = op_script(data)
            expected = image.rgb_to_bgr(data)
            assert_allclose(actual, expected)

    def test_batch_rgb_to_bgr(self):

        # prepare input data
        data = torch.tensor([[[[1., 1.],
                               [1., 1.]],

                              [[2., 2.],
                               [2., 2.]],

                              [[3., 3.],
                               [3., 3.]]],

                             [[[1., 1.],
                               [1., 1.]],

                              [[2., 2.],
                               [2., 2.]],

                              [[3., 3.],
                               [3., 3.]]]])  # 2x3x2x2

        expected = torch.tensor([[[[3., 3.],
                                   [3., 3.]],

                                  [[2., 2.],
                                   [2., 2.]],

                                  [[1., 1.],
                                   [1., 1.]]],

                                 [[[3., 3.],
                                   [3., 3.]],

                                  [[2., 2.],
                                   [2., 2.]],

                                  [[1., 1.],
                                   [1., 1.]]]])  # 2x3x2x2

        f = image.RgbToBgr()
        out = f(data)
        assert_allclose(out, expected)


class TestRgbToGrayscale:
    def test_rgb_to_grayscale(self):
        channels, height, width = 3, 4, 5
        img = torch.ones(channels, height, width)
        assert image.RgbToGrayscale()(img).shape == (1, height, width)

    def test_rgb_to_grayscale_batch(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        assert image.RgbToGrayscale()(img).shape == \
            (batch_size, 1, height, width)

    def test_gradcheck(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(image.rgb_to_grayscale, (img,), raise_exception=True)

    def test_jit(self):
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width)
        gray = image.RgbToGrayscale()
        gray_traced = torch.jit.trace(image.RgbToGrayscale(), img)
        assert_allclose(gray(img), gray_traced(img))
