import pytest
import torch

import kornia

from testing.base import BaseTester


class TestLambdaModule(BaseTester):
    def add_2_layer(self, tensor):
        return tensor + 2

    def add_x_mul_y(self, tensor, x, y=2):
        return torch.mul(tensor + x, y)

    def test_smoke(self, device, dtype):
        B, C, H, W = 1, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        func = self.add_2_layer
        if not callable(func):
            raise TypeError(f"Argument lambd should be callable, got {type(func).__name__!r}")
        assert isinstance(kornia.contrib.Lambda(func)(img), torch.Tensor)

    @pytest.mark.parametrize("x", [3, 2, 5])
    def test_lambda_with_arguments(self, x, device, dtype):
        B, C, H, W = 2, 3, 5, 7
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        func = self.add_x_mul_y
        lambda_module = kornia.contrib.Lambda(func)
        out = lambda_module(img, x)
        assert isinstance(out, torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 2, 3), (2, 3, 5, 7)])
    def test_lambda(self, shape, device, dtype):
        B, C, H, W = shape
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        func = kornia.color.bgr_to_grayscale
        lambda_module = kornia.contrib.Lambda(func)
        out = lambda_module(img)
        assert isinstance(out, torch.Tensor)

    def test_gradcheck(self, device):
        B, C, H, W = 1, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        func = kornia.color.bgr_to_grayscale
        self.gradcheck(kornia.contrib.Lambda(func), (img,))
