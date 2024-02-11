from kornia.core import tensor
from kornia.utils.misc import (
    differentiable_clipping,
    differentiable_polynomial_floor,
    differentiable_polynomial_rounding,
)

from testing.base import BaseTester


class TestDifferentiableClipping(BaseTester):
    def test_differentiable_clipping(self, device):
        x = tensor([1.0, 6.0, 10.0, 12.0], device=device)
        y = differentiable_clipping(x, min=5.0, max=10.0)
        y_expected = tensor([4.9804, 6.0, 10.0, 10.0173], device=device)

        self.assert_close(y, y_expected)


class TestDifferentiablePolynomialRounding(BaseTester):
    def test_differentiable_polynomial_rounding(self, device):
        x = tensor([1.5], device=device)
        y = differentiable_polynomial_rounding(x)
        y_expected = tensor([1.875], device=device)

        self.assert_close(y, y_expected)


class TestDifferentiablePolynomialFloor(BaseTester):
    def test_differentiable_polynomial_floor(self, device):
        x = tensor([1.5], device=device)
        y = differentiable_polynomial_floor(x)
        y_expected = tensor([1.0], device=device)

        self.assert_close(y, y_expected)
