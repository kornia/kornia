import pytest
import torch

from kornia.filters import (
    GaussianBlur2d,
    gaussian,
    gaussian_blur2d,
    get_gaussian_discrete_kernel1d,
    get_gaussian_erf_kernel1d,
    get_gaussian_kernel1d,
    get_gaussian_kernel2d,
    get_gaussian_kernel3d,
)

from testing.base import BaseTester, assert_close


@pytest.mark.parametrize(
    "window_size, sigma, mean, expected",
    [
        (5, 1.0, None, torch.tensor([[0.0545, 0.2442, 0.4026, 0.2442, 0.0545]])),
        (
            11,
            5.0,
            None,
            [[0.0663, 0.0794, 0.0914, 0.1010, 0.1072, 0.1094, 0.1072, 0.1010, 0.0914, 0.0794, 0.0663]],
        ),
        (
            11,
            5.0,
            8.0,
            [[0.0343, 0.0463, 0.0600, 0.0747, 0.0895, 0.1029, 0.1138, 0.1208, 0.1232, 0.1208, 0.1138]],
        ),
        (
            11,
            11.0,
            3.0,
            [[0.0926, 0.0946, 0.0957, 0.0961, 0.0957, 0.0946, 0.0926, 0.0900, 0.0867, 0.0828, 0.0785]],
        ),
    ],
)
def test_gaussian(window_size, sigma, mean, expected, device, dtype):
    expected = torch.tensor(expected, device=device, dtype=dtype)
    result = gaussian(window_size, sigma, mean=mean, device=device, dtype=dtype)
    assert_close(result, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [1.5, 5.0])
def test_get_gaussian_kernel1d_float(window_size, sigma, device, dtype):
    actual = get_gaussian_kernel1d(window_size, sigma, device=device, dtype=dtype)
    expected = torch.ones(1, device=device, dtype=dtype)

    assert actual.shape == (1, window_size)
    assert_close(actual.sum(), expected.sum())


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [[[1.5]], [[1.5], [5.0]], [[1.5], [5.0]]])
def test_get_gaussian_kernel1d_tensor(window_size, sigma, device, dtype):
    sigma = torch.tensor(sigma, device=device, dtype=dtype)
    bs = sigma.shape[0]

    actual = get_gaussian_kernel1d(window_size, sigma)
    expected = torch.ones(bs, device=device, dtype=dtype)

    assert actual.shape == (bs, window_size)
    assert_close(actual.sum(), expected.sum())


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize("sigma", [(1.5, 1.5), (2.1, 2.1)])
def test_get_gaussian_kernel2d_float(ksize_x, ksize_y, sigma, device, dtype):
    actual = get_gaussian_kernel2d((ksize_y, ksize_x), sigma, device=device, dtype=dtype)
    expected = torch.ones(1, device=device, dtype=dtype)

    assert actual.shape == (1, ksize_y, ksize_x)
    assert_close(actual.sum(), expected.sum())


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize("sigma", ([[1.5, 2.1], [1.5, 2.1], [5.0, 2.7]], [[1.5, 2.1], [3.5, 2.1]]))
def test_get_gaussian_kernel2d_tensor(ksize_x, ksize_y, sigma, device, dtype):
    sigma = torch.tensor(sigma, device=device, dtype=dtype)
    bs = sigma.shape[0]

    actual = get_gaussian_kernel2d((ksize_y, ksize_x), sigma)
    expected = torch.ones(bs, device=device, dtype=dtype)

    assert actual.shape == (bs, ksize_y, ksize_x)
    assert_close(actual.sum(), expected.sum())


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize("ksize_z", [9, 3])
@pytest.mark.parametrize("sigma", [(1.5, 1.5, 3.5), (2.1, 1.5, 2.1)])
def test_get_gaussian_kernel3d_float(ksize_x, ksize_y, ksize_z, sigma, device, dtype):
    actual = get_gaussian_kernel3d((ksize_z, ksize_y, ksize_x), sigma, device=device, dtype=dtype)
    expected = torch.ones(1, device=device, dtype=dtype)

    assert actual.shape == (1, ksize_z, ksize_y, ksize_x)
    assert_close(actual.sum(), expected.sum())


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize("ksize_z", [9, 3])
@pytest.mark.parametrize(
    "sigma", ([[1.5, 2.1, 3.5], [1.5, 2.1, 1.5], [5.0, 2.7, 2.1]], [[1.5, 3.5, 2.1], [1.2, 3.5, 2.1]])
)
def test_get_gaussian_kernel3d_tensor(ksize_x, ksize_y, ksize_z, sigma, device, dtype):
    sigma = torch.tensor(sigma, device=device, dtype=dtype)
    bs = sigma.shape[0]

    actual = get_gaussian_kernel3d((ksize_z, ksize_y, ksize_x), sigma)
    expected = torch.ones(bs, device=device, dtype=dtype)

    assert actual.shape == (bs, ksize_z, ksize_y, ksize_x)
    assert_close(actual.sum(), expected.sum())


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [1.5, 5.0])
def test_get_discrete_gaussian_erf_kernel1d_float(window_size, sigma, device, dtype):
    actual = get_gaussian_erf_kernel1d(window_size, sigma, device=device, dtype=dtype)
    expected = torch.ones(1, device=device, dtype=dtype)

    assert actual.shape == (1, window_size)
    assert_close(actual.sum(), expected.sum())


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [[[1.5]], [[1.5], [5.0]], [[1.5], [5.0]]])
def test_get_discrete_gaussian_erf_kernel1d_tensor(window_size, sigma, device, dtype):
    sigma = torch.tensor(sigma, device=device, dtype=dtype)
    bs = sigma.shape[0]

    actual = get_gaussian_erf_kernel1d(window_size, sigma)
    expected = torch.ones(bs, device=device, dtype=dtype)

    assert actual.shape == (bs, window_size)
    assert_close(actual.sum(), expected.sum())


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [1.5, 5.0])
def test_get_gaussian_discrete_kernel1d_float(window_size, sigma, device, dtype):
    actual = get_gaussian_discrete_kernel1d(window_size, sigma, device=device, dtype=dtype)
    expected = torch.ones(1, device=device, dtype=dtype)

    assert actual.shape == (1, window_size)
    assert_close(actual.sum(), expected.sum())


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("sigma", [[[1.5]], [[1.5], [5.0]], [[1.5], [5.0]]])
def test_get_gaussian_discrete_kernel1d_tensor(window_size, sigma, device, dtype):
    sigma = torch.tensor(sigma, device=device, dtype=dtype)
    bs = sigma.shape[0]

    actual = get_gaussian_discrete_kernel1d(window_size, sigma)
    expected = torch.ones(bs, device=device, dtype=dtype)

    assert actual.shape == (bs, window_size)
    assert_close(actual.sum(), expected.sum())


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize("sigma", [(1.5, 1.5), (2.1, 2.1)])
def test_gaussian_blur2d_float(ksize_x, ksize_y, sigma, device, dtype):
    sample = torch.rand(1, 3, 16, 16, device=device, dtype=dtype)

    actual = gaussian_blur2d(sample, (ksize_y, ksize_x), sigma, "replicate", separable=False)
    actual_sep = gaussian_blur2d(sample, (ksize_y, ksize_x), sigma, "replicate", separable=True)

    assert_close(actual, actual_sep)


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
@pytest.mark.parametrize("sigma", ([[1.5, 2.1], [1.5, 2.1], [5.0, 2.7]], [[1.5, 2.1], [3.5, 2.1]]))
def test_gaussian_blur2d_tensor(ksize_x, ksize_y, sigma, device, dtype):
    sigma = torch.tensor(sigma, device=device, dtype=dtype)
    bs = sigma.shape[0]
    sample = torch.rand(bs, 3, 16, 16, device=device, dtype=dtype)
    actual = gaussian_blur2d(sample, (ksize_y, ksize_x), sigma, "replicate", separable=False)
    actual_sep = gaussian_blur2d(sample, (ksize_y, ksize_x), sigma, "replicate", separable=True)

    assert_close(actual, actual_sep)


class TestGaussianBlur2d(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [3, (5, 5), (5, 7)])
    @pytest.mark.parametrize("separable", [False, True])
    def test_smoke(self, shape, kernel_size, separable, device, dtype):
        B, C, H, W = shape
        data = torch.rand(B, C, H, W, device=device, dtype=dtype)
        sigma_tensor = torch.rand(B, 2, device=device, dtype=dtype)

        actual_A = gaussian_blur2d(data, kernel_size, sigma_tensor, "reflect", separable)
        assert isinstance(actual_A, torch.Tensor)
        assert actual_A.shape == shape

        sigma = tuple(sigma_tensor[0, ...].cpu().numpy().tolist())
        actual_B = gaussian_blur2d(data, kernel_size, sigma, "reflect", separable)
        assert isinstance(actual_B, torch.Tensor)
        assert actual_B.shape == shape

        # Just the first item of the batch use the same sigma
        self.assert_close(actual_A[0, ...], actual_B[0, ...])

    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [3, (5, 5), (5, 7)])
    def test_cardinality(self, shape, kernel_size, device, dtype):
        sigma = (1.5, 2.1)
        sample = torch.rand(shape, device=device, dtype=dtype)
        actual = gaussian_blur2d(sample, kernel_size, sigma, "replicate")
        assert actual.shape == shape

    def test_exception(self):
        # input should be a tensor
        with pytest.raises(Exception) as errinfo:
            gaussian_blur2d(1, 3, (1.0, 1.0))
        assert "Not a Tensor type. Go" in str(errinfo)

        # Sigma should be a tuple or a tensor
        with pytest.raises(Exception) as errinfo:
            gaussian_blur2d(torch.rand(1, 1, 1, 1), 3, 1.0)
        assert "Not a Tensor type. Go" in str(errinfo)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        sample = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = (3, 3)
        sigma = (1.5, 2.1)
        actual = gaussian_blur2d(sample, kernel_size, sigma, "replicate")
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        # test parameters
        batch_shape = (1, 3, 5, 5)
        kernel_size = (3, 3)
        sigma = (1.5, 2.1)

        # evaluate function gradient
        sample = torch.rand(batch_shape, device=device, dtype=torch.float64)
        self.gradcheck(gaussian_blur2d, (sample, kernel_size, sigma, "replicate"))

    @pytest.mark.parametrize("kernel_size", [3, (5, 5), (5, 7)])
    @pytest.mark.parametrize("sigma", [(1.5, 2.1), (0.5, 0.5)])
    def test_module(self, kernel_size, sigma, device, dtype):
        params = [kernel_size, sigma]
        op = gaussian_blur2d
        op_module = GaussianBlur2d(*params)

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)

        self.assert_close(op(img, *params), op_module(img))
        sigma_tensor = torch.tensor([sigma], device=device, dtype=dtype)
        params = [kernel_size, sigma_tensor]
        op_module = GaussianBlur2d(*params)

        self.assert_close(op(img, *params), op_module(img))

    @pytest.mark.parametrize("kernel_size", [3, (5, 5), (5, 7)])
    @pytest.mark.parametrize("sigma", [(1.5, 2.1), (0.5, 0.5)])
    def test_dynamo(self, kernel_size, sigma, device, dtype, torch_optimizer):
        data = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)

        op = GaussianBlur2d(kernel_size, sigma)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))

        sigma_tensor = torch.tensor([sigma], device=device, dtype=dtype)
        op = GaussianBlur2d(kernel_size, sigma_tensor)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))
