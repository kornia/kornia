from kornia.core import Tensor
from kornia.utils.misc import (
    differentiable_clipping,
    differentiable_polynomial_rounding,
    differentiable_polynomial_floor
)

import torch

class TestDifferentiableClipping:
    def test_differentiable_clipping(self, device):
        x = Tensor([1.0])
        y = differentiable_clipping(x, min=10.0)
        y_expected = Tensor([9.9800])

        assert(y, y_expected)

class TestDifferentiablePolynomialRounding:
    def test_differentiable_polynomial_rounding(self, device):
        x = Tensor([1.0])
        y = differentiable_polynomial_rounding(x)
        y_expected = torch.round(x) + (x - torch.round(x)) ** 3

        assert(y, y_expected)
    
class TestDifferentiablePolynomialFloor:
    def test_differentiable_polynomial_floor(self, device):
        x = Tensor([1.0])
        y = differentiable_polynomial_floor(x)
        y_expected = torch.floor(x) + (x - 0.5 - torch.floor(x)) ** 3

        assert(y, y_expected)