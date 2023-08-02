import pytest
import torch
from torch.autograd import gradcheck

import kornia.geometry.solvers as solver
from kornia.testing import assert_close


class TestQuadraticSolver:
    def test_smoke(self, device, dtype):
        coeffs = torch.rand(1, 3, device=device, dtype=dtype)
        roots = solver.solve_quadratic(coeffs)
        assert roots.shape == (1, 2)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 7])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        coeffs = torch.rand(B, 3, device=device, dtype=dtype)
        roots = solver.solve_quadratic(coeffs)
        assert roots.shape == (B, 2)

    @pytest.mark.parametrize(
        "coeffs, expected_solutions",
        [
            (torch.tensor([[1.0, 4.0, 4.0]]), torch.tensor([[-2.0, -2.0]])),  # zero discriminant
            (torch.tensor([[1.0, -5.0, 6.0]]), torch.tensor([[3.0, 2.0]])),
            (torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[0.0, 0.0]])),  # negative discriminant
        ],
    )
    def test_solve_quadratic(self, coeffs, expected_solutions, device, dtype):
        roots = solver.solve_quadratic(coeffs)
        assert_close(roots[0], expected_solutions[0])

    def gradcheck(self, device):
        coeffs = torch.rand(1, 3, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(solver.solve_quadratic, (coeffs), raise_exception=True, fast_mode=True)


class TestCubicSolver:
    def test_smoke(self, device, dtype):
        coeffs = torch.rand(1, 4, device=device, dtype=dtype)
        roots = solver.solve_cubic(coeffs)
        assert roots.shape == (1, 3)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 7])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        coeffs = torch.rand(B, 4, device=device, dtype=dtype)
        roots = solver.solve_cubic(coeffs)
        assert roots.shape == (B, 3)

    @pytest.mark.parametrize(
        "coeffs, expected_solutions",
        [
            (torch.tensor([[2.0, 3.0, -11.0, -6.0]]), torch.tensor([[2.0, -3.0, -0.5]])),
            (torch.tensor([[1.0, 0.0, 4.0, 4.0]]), torch.tensor([[-0.847, 0.0, 0.0]])),
            (torch.tensor([[2.0, -6.0, 6.0, -2.0]]), torch.tensor([[1.0, 1.0, 1.0]])),
            (torch.tensor([[0.0, 0.0, 3.0, -1.0]]), torch.tensor([[1.0, 0.0, 0.0]])),  # handle first order
            (torch.tensor([[0.0, 1.0, -5.0, 6.0]]), torch.tensor([[3.0, 2.0, 0.0]])),  # handle second order
        ],
    )
    def test_solve_quadratic(self, coeffs, expected_solutions, device, dtype):
        roots = solver.solve_cubic(coeffs)
        assert_close(roots[0], expected_solutions[0], rtol=1e-3, atol=1e-3)

    def gradcheck(self, device):
        coeffs = torch.rand(1, 4, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(solver.solve_cubic, (coeffs), raise_exception=True, fast_mode=True)
