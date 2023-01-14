import pytest
import torch

from kornia.enhance.integral import IntegralImage, IntegralTensor, integral_image, integral_tensor
from kornia.testing import BaseTester


class TestIntegralTensor(BaseTester):
    def test_smoke(self, device, dtype):
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        output = integral_tensor(tensor)
        assert output.shape == (1, 1, 4, 4)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1)])
    def test_cardinality(self, device, dtype, shape):
        tensor = torch.rand(*shape, device=device, dtype=dtype)
        output = integral_tensor(tensor)
        assert output.shape == shape

    def test_exception(self, device, dtype):
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        with pytest.raises(Exception):
            dim = (0, 1, 2, 3, 4)
            integral_tensor(tensor, dim)
        with pytest.raises(Exception):
            dim = (4, 5)
            integral_tensor(tensor, dim)

    def test_module(self, device, dtype):
        module = IntegralTensor()
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        output = module(tensor)
        assert output.shape == (1, 1, 4, 4)

    def test_gradcheck(self, device):
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(integral_tensor, (tensor,), raise_exception=True)

    def test_value(self, device, dtype):
        tensor = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], device=device, dtype=dtype)
        output = integral_tensor(tensor)
        expected = torch.tensor([[[[1.0, 3.0, 6.0], [4.0, 9.0, 15.0], [7.0, 15.0, 24.0]]]], device=device, dtype=dtype)
        self.assert_close(output, expected)


class TestIntegralImage(BaseTester):
    def test_smoke(self, device, dtype):
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        output = integral_image(tensor)
        assert output.shape == (1, 1, 4, 4)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1)])
    def test_cardinality(self, device, dtype, shape):
        tensor = torch.rand(*shape, device=device, dtype=dtype)
        output = integral_image(tensor)
        assert output.shape == shape

    def test_exception(self, device, dtype):
        tensor = torch.rand(4, device=device, dtype=dtype)
        with pytest.raises(Exception):
            integral_image(tensor)

    def test_module(self, device, dtype):
        module = IntegralImage()
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        output = module(tensor)
        assert output.shape == (1, 1, 4, 4)

    def test_gradcheck(self, device):
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(integral_image, (tensor,), raise_exception=True)

    def test_values(self, device, dtype):
        tensor = torch.tensor(
            [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], device=device, dtype=dtype
        )
        output = integral_image(tensor)
        expected = torch.tensor(
            [[[[1, 3, 6, 10], [6.0, 14.0, 24.0, 36.0], [15.0, 33.0, 54.0, 78.0], [28.0, 60.0, 96.0, 136.0]]]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(output, expected)
