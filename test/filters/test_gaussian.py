import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [1.5, 5.0])
def test_get_gaussian_kernel(window_size, sigma):
    kernel = kornia.filters.get_gaussian_kernel1d(window_size, sigma)
    assert kernel.shape == (1, window_size)
    assert kernel.sum().item() == pytest.approx(1.0)


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
@pytest.mark.parametrize("sigma", [1.5, 2.1])
def test_get_gaussian_kernel2d(ksize_x, ksize_y, sigma):
    kernel = kornia.filters.get_gaussian_kernel2d((ksize_x, ksize_y), (sigma, sigma))
    assert kernel.shape == (1, ksize_x, ksize_y)
    assert kernel.sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize("sigma", [1.5, 2.1])
def test_separable(ksize_x, ksize_y, sigma, device, dtype):
    input = torch.rand(2, 3, 16, 16, device=device, dtype=dtype)
    out = kornia.filters.gaussian_blur2d(input, (ksize_x, ksize_y), (sigma, sigma), "replicate", separable=False)
    out_sep = kornia.filters.gaussian_blur2d(input, (ksize_x, ksize_y), (sigma, sigma), "replicate", separable=True)

    assert_close(out, out_sep)


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [torch.tensor([1.5, 5.0]), torch.tensor([1.5, 5.0])])
def test_get_gaussian_kernel_t(window_size, sigma):
    kernel = kornia.filters.get_gaussian_kernel1d_t(window_size, sigma)
    assert kernel.shape == (2, window_size)
    assert_close(kernel.sum(1), torch.ones(2))


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize("sigma", [torch.tensor([[1.5, 2.1], [1.5, 2.1]]), torch.tensor([[1.5, 2.1], [3.5, 2.1]])])
def test_get_gaussian_kernel2d_t(ksize_x, ksize_y, sigma):
    kernel = kornia.filters.get_gaussian_kernel2d_t((ksize_x, ksize_y), sigma)
    assert kernel.shape == (2, ksize_x, ksize_y)
    assert_close(kernel.sum([1, 2]), torch.ones(2))


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize("sigma", [torch.tensor([[1.5, 2.1], [1.5, 2.1]]), torch.tensor([[1.5, 2.1], [2.5, 2.1]])])
def test_separable_t(ksize_x, ksize_y, sigma, device, dtype):
    input = torch.rand(2, 3, 16, 16, device=device, dtype=dtype)
    out = kornia.filters.gaussian_blur2d_t(input, (ksize_x, ksize_y), sigma, "replicate", separable=False)
    out_sep = kornia.filters.gaussian_blur2d_t(input, (ksize_x, ksize_y), sigma, "replicate", separable=True)

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

    def test_module_jit(self, device, dtype):
        params = [(3, 3), (1.5, 1.5)]
        op_module = kornia.filters.GaussianBlur2d(*params)
        op_script = torch.jit.script(op_module)

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        assert_close(op_module(img), op_script(img))

    def test_module_change_param(self, device, dtype):
        """Module should reflect changes to properties."""
        orig_params = [(3, 3), (1.5, 1.5)]
        new_params = [(1, 5), (0.2, 2.4)]
        op = kornia.filters.gaussian_blur2d
        op_module = kornia.filters.GaussianBlur2d(*orig_params)

        assert op_module.kernel_size == orig_params[0]
        assert op_module.sigma == orig_params[1]

        op_module.kernel_size = new_params[0]
        op_module.sigma = new_params[1]
        assert op_module.kernel_size == new_params[0]
        assert op_module.sigma == new_params[1]

        img = torch.rand(1, 3, 5, 5, device=device, dtype=dtype)
        assert_close(op(img, *new_params), op_module(img))

    def test_module_change_separable(self, device, dtype):
        """Module should clear out unused kernel when switching separable to non separable."""
        params = [(3, 3), (1.5, 1.5)]
        op = kornia.filters.gaussian_blur2d
        op_module = kornia.filters.GaussianBlur2d(*params, separable=True)
        assert op_module.separable

        assert list(op_module.kernel_x.shape) == [3]
        assert list(op_module.kernel_x.shape) == [3]
        op_module.separable = False
        assert not op_module.separable
        assert not getattr(op_module, "kernel_x", None)

        img = torch.rand(1, 3, 5, 5, device=device, dtype=dtype)
        assert_close(op(img, *params), op_module(img))
