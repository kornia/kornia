import pytest
import torch

from kornia.filters import BilateralBlur, bilateral_blur
from kornia.testing import BaseTester, tensor_to_gradcheck_var


class TestBilateralBlur(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 1, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    @pytest.mark.parametrize("sigma_color", [1, 0.1])
    @pytest.mark.parametrize("sigma_space", [(1, 1), (1.5, 1)])
    @pytest.mark.parametrize("color_distance_type", ["l1", "l2"])
    def test_smoke(self, shape, kernel_size, sigma_color, sigma_space, color_distance_type, device, dtype):
        inp = torch.zeros(shape, device=device, dtype=dtype)
        actual = bilateral_blur(inp, kernel_size, sigma_color, sigma_space, color_distance_type=color_distance_type)
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == shape

    @pytest.mark.parametrize("shape", [(1, 1, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    def test_cardinality(self, shape, kernel_size, device, dtype):
        inp = torch.zeros(shape, device=device, dtype=dtype)
        actual = bilateral_blur(inp, kernel_size, 0.1, (1, 1))
        assert actual.shape == shape

    def test_exception(self):
        with pytest.raises(Exception) as errinfo:
            bilateral_blur(torch.rand(1, 1, 5, 5), 3, 1, 1)
        assert 'Not a Tensor type. Go' in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            bilateral_blur(torch.rand(1, 1, 5, 5), 3, 0.1, (1, 1), color_distance_type="l3")
        assert 'color_distance_type only acceps l1 or l2' in str(errinfo)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = bilateral_blur(inp, 3, 1, (1, 1))
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        img = torch.rand(1, 2, 5, 4, device=device)
        img = tensor_to_gradcheck_var(img)  # to var
        self.gradcheck(bilateral_blur, (img, 3, 1, (1, 1)))

    @pytest.mark.parametrize("shape", [(1, 1, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    @pytest.mark.parametrize("sigma_color", [1, 0.1])
    @pytest.mark.parametrize("sigma_space", [(1, 1), (1.5, 1)])
    @pytest.mark.parametrize("color_distance_type", ["l1", "l2"])
    def test_module(self, shape, kernel_size, sigma_color, sigma_space, color_distance_type, device, dtype):
        img = torch.rand(shape, device=device, dtype=dtype)
        params = (kernel_size, sigma_color, sigma_space, "reflect", color_distance_type)

        op = bilateral_blur
        op_module = BilateralBlur(*params)
        self.assert_close(op_module(img), op(img, *params))

    @pytest.mark.parametrize("shape", [(1, 3, 8, 15), (2, 1, 11, 7)])
    @pytest.mark.parametrize('kernel_size', [5, (5, 7)])
    @pytest.mark.parametrize('color_distance_type', ["l1", "l2"])
    def test_dynamo(self, shape, kernel_size, color_distance_type, device, dtype, torch_optimizer):
        inpt = torch.ones(shape, device=device, dtype=dtype)
        op = BilateralBlur(kernel_size, 1, (1, 1), color_distance_type=color_distance_type)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt), op_optimized(inpt))
