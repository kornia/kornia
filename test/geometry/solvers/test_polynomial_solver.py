import pytest
import torch
from torch.autograd import gradcheck

import kornia.geometry.solvers as solver
import kornia.testing as utils2
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

    @pytest.mark.parametrize("coeffs, expected_solutions", [
        (torch.tensor([[1, 4, 4]]), torch.tensor([[-2, -2]])), # zero discriminant
        (torch.tensor([[1, -5, 6]]), torch.tensor([[3, 2]])), 
        (torch.tensor([[1, 2, 3]]), torch.tensor([[0, 0]])), # negative discriminant
        ])
    def test_solve_quadratic(self, coeffs, expected_solutions, device, dtype):
        roots = solver.solve_quadratic(coeffs)
        assert_close(roots[0], expected_solutions[0])

