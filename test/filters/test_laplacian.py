import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


@pytest.mark.parametrize("window_size", [5])
def test_get_laplacian_kernel(window_size):
    kernel = kornia.filters.get_laplacian_kernel1d(window_size)
    assert kernel.shape == (window_size,)
    assert kernel.sum().item() == pytest.approx(0.0)


@pytest.mark.parametrize("window_size", [7])
def test_get_laplacian_kernel2d(window_size):
    kernel = kornia.filters.get_laplacian_kernel2d(window_size)
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
        ],
        device=kernel.device,
    )
    assert_close(expected, kernel)


class TestLaplacian:
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_cardinality(self, batch_shape, device, dtype):
        kernel_size = 5

        input = torch.rand(batch_shape, device=device, dtype=dtype)
        actual = kornia.filters.laplacian(input, kernel_size)
        assert actual.shape == batch_shape

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        input = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = 3
        actual = kornia.filters.laplacian(input, kernel_size)
        assert_close(actual, actual)

    def test_gradcheck(self, device, dtype):
        # test parameters
        batch_shape = (1, 2, 5, 7)
        kernel_size = 3

        # evaluate function gradient
        input = torch.rand(batch_shape, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)
        assert gradcheck(kornia.filters.laplacian, (input, kernel_size), raise_exception=True, fast_mode=True)

    def test_module(self, device, dtype):
        params = [3]
        op = kornia.filters.laplacian
        op_module = kornia.filters.Laplacian(*params)

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        assert_close(op(img, *params), op_module(img))
