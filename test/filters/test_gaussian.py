import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [1.5, 5.0, torch.tensor([1.5, 5.0]), torch.tensor([1.5, 5.0])])
def test_get_gaussian_kernel(window_size, sigma):
    kernel = kornia.filters.get_gaussian_kernel1d(window_size, sigma)
    if isinstance(sigma, torch.Tensor):
        bs = sigma.shape[0]
    else:
        bs = 1
    assert kernel.shape == (bs, window_size)

    expected = torch.ones(bs, device=kernel.device)
    assert_close(kernel.sum(), expected.sum())


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [1.5, 5.0])
def test_get_gaussian_discrete_kernel(window_size, sigma):
    kernel = kornia.filters.get_gaussian_discrete_kernel1d(window_size, sigma)
    assert kernel.shape == (window_size,)
    assert kernel.sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [1.5, 5.0])
def test_get_gaussian_erf_kernel(window_size, sigma):
    kernel = kornia.filters.get_gaussian_erf_kernel1d(window_size, sigma)
    assert kernel.shape == (window_size,)
    assert kernel.sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize(
    "sigma", [(1.5, 1.5), (2.1, 2.1), torch.tensor([[1.5, 2.1], [1.5, 2.1]]), torch.tensor([[1.5, 2.1], [3.5, 2.1]])]
)
def test_get_gaussian_kernel2d(ksize_x, ksize_y, sigma):
    kernel = kornia.filters.get_gaussian_kernel2d((ksize_x, ksize_y), sigma)

    if isinstance(sigma, tuple):
        bs = 1
    else:
        bs = sigma.shape[0]

    assert kernel.shape == (bs, ksize_x, ksize_y)

    expected = torch.ones(bs, device=kernel.device)
    assert_close(kernel.sum(), expected.sum())


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize(
    "sigma", [1.5, 2.1, torch.tensor([[1.5, 2.1], [1.5, 2.1]]), torch.tensor([[1.5, 2.1], [2.5, 2.1]])]
)
def test_separable(ksize_x, ksize_y, sigma, device, dtype):
    input = torch.rand(2, 3, 16, 16, device=device, dtype=dtype)
    if isinstance(sigma, float):
        sigma = (sigma, sigma)

    out = kornia.filters.gaussian_blur2d(input, (ksize_x, ksize_y), sigma, "replicate", separable=False)
    out_sep = kornia.filters.gaussian_blur2d(input, (ksize_x, ksize_y), sigma, "replicate", separable=True)

    assert_close(out, out_sep)


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
        assert_close(actual, actual)

    def test_gradcheck(self, device, dtype):
        # test parameters
        batch_shape = (1, 3, 5, 5)
        kernel_size = (3, 3)
        sigma = (1.5, 2.1)

        # evaluate function gradient
        input = torch.rand(batch_shape, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(
            kornia.filters.gaussian_blur2d,
            (input, kernel_size, sigma, "replicate"),
            raise_exception=True,
            fast_mode=True,
        )

    @pytest.mark.skip(reason="`kornia.utils.get_cuda_device_if_available` is not jittable")
    def test_jit(self, device, dtype):
        op = kornia.filters.gaussian_blur2d
        op_script = torch.jit.script(op)
        params = [(3, 3), (1.5, 1.5)]

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        assert_close(op(img, *params), op_script(img, *params))

    def test_module(self, device, dtype):
        params = [(3, 3), (1.5, 1.5)]
        op = kornia.filters.gaussian_blur2d
        op_module = kornia.filters.GaussianBlur2d(*params)

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        assert_close(op(img, *params), op_module(img))
