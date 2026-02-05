# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import torch

import kornia.geometry.solvers as solver

from testing.base import BaseTester


class TestQuadraticSolver(BaseTester):
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
        self.assert_close(roots[0], expected_solutions[0])

    def gradcheck(self, device):
        coeffs = torch.rand(1, 3, device=device, dtype=torch.float64, requires_grad=True)
        assert self.gradcheck(solver.solve_quadratic, (coeffs), raise_exception=True, fast_mode=True)


class TestCubicSolver(BaseTester):
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
            (torch.tensor([[0.0, 0.0, 1.0, -1.0]]), torch.tensor([[1.0, 0.0, 0.0]])),  # handle first order
            (torch.tensor([[0.0, 1.0, -5.0, 6.0]]), torch.tensor([[3.0, 2.0, 0.0]])),  # handle second order
        ],
    )
    def test_solve_quadratic_in_cubic(self, coeffs, expected_solutions, device, dtype):
        roots = solver.solve_cubic(coeffs)
        self.assert_close(roots[0], expected_solutions[0], rtol=1e-3, atol=1e-3)

    def gradcheck(self, device):
        coeffs = torch.rand(1, 4, device=device, dtype=torch.float64, requires_grad=True)
        assert self.gradcheck(solver.solve_cubic, (coeffs), raise_exception=True, fast_mode=True)


class TestMultiplyDegOnePoly(BaseTester):
    def test_smoke(self, device, dtype):
        a = torch.rand(1, 4, device=device, dtype=dtype)
        b = torch.rand(1, 4, device=device, dtype=dtype)
        out_poly = solver.multiply_deg_one_poly(a, b)
        assert out_poly.shape == (1, 10)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        a = torch.rand(B, 4, device=device, dtype=dtype)
        b = torch.rand(B, 4, device=device, dtype=dtype)
        out_poly = solver.multiply_deg_one_poly(a, b)
        assert out_poly.shape == (B, 10)

    @pytest.mark.parametrize(
        "a_coeffs, b_coeffs, expected_coeffs",
        [
            # Case 1: (x + 2y + 3z + 4) * (5x + 6y + 7z + 8)
            (
                torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
                torch.tensor([[5.0, 16.0, 22.0, 28.0, 12.0, 32.0, 40.0, 21.0, 52.0, 32.0]]),
            ),
            # Case 2: Squaring a polynomial (x - y + 2z - 3)^2
            (
                torch.tensor([[1.0, -1.0, 2.0, -3.0]]),
                torch.tensor([[1.0, -1.0, 2.0, -3.0]]),
                torch.tensor([[1.0, -2.0, 4.0, -6.0, 1.0, -4.0, 6.0, 4.0, -12.0, 9.0]]),
            ),
            # Case 3: Multiplying by zero
            (
                torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
                torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            ),
            # Case 4: Only constant terms (10) * (5)
            (
                torch.tensor([[0.0, 0.0, 0.0, 10.0]]),
                torch.tensor([[0.0, 0.0, 0.0, 5.0]]),
                torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]),
            ),
        ],
    )
    def test_values(self, a_coeffs, b_coeffs, expected_coeffs, device, dtype):
        # Move tensor data to the target device and dtype
        a = a_coeffs.to(device, dtype)
        b = b_coeffs.to(device, dtype)
        expected = expected_coeffs.to(device, dtype)

        # Compute the result
        result = solver.multiply_deg_one_poly(a, b)

        # Compare result with expected values
        self.assert_close(result, expected, rtol=1e-4, atol=1e-4)


class TestMultiplyDegTwoOnePoly(BaseTester):
    def test_smoke(self, device, dtype):
        a = torch.rand(1, 10, device=device, dtype=dtype)
        b = torch.rand(1, 4, device=device, dtype=dtype)
        out_poly = solver.multiply_deg_two_one_poly(a, b)
        assert out_poly.shape == (1, 20)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        a = torch.rand(B, 10, device=device, dtype=dtype)
        b = torch.rand(B, 4, device=device, dtype=dtype)
        out_poly = solver.multiply_deg_two_one_poly(a, b)
        assert out_poly.shape == (B, 20)

    @pytest.mark.parametrize(
        "a_coeffs, b_coeffs, expected_coeffs",
        [
            # Case 1: (x^2 + 2y) * (3x + 4) = 3x^3 + 4x^2 + 6xy + 8y
            (
                torch.tensor([[1.0, 0, 0, 0, 0, 0, 2.0, 0, 0, 0]]),
                torch.tensor([[3.0, 0, 0, 4.0]]),
                torch.tensor([[3.0, 0, 0, 0, 0, 4.0, 0, 0, 0, 6.0, 0, 0, 0, 0, 0, 8.0, 0, 0, 0, 0]]),
            ),
            # Case 2: (xy + z^2) * (y + z) = xy^2 + xyz + yz^2 + z^3
            (
                torch.tensor([[0, 1.0, 0, 0, 0, 0, 0, 1.0, 0, 0]]),
                torch.tensor([[0, 1.0, 1.0, 0]]),
                torch.tensor([[0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0]]),
            ),
            # Case 3: Multiplying a complex polynomial by a constant: (x^2+y) * 5
            (
                torch.tensor([[1.0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0]]),
                torch.tensor([[0, 0, 0, 5.0]]),
                # Expected: 5x^2 + 5y
                torch.tensor([[0, 0, 0, 0, 0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0, 0, 0, 0, 0]]),
            ),
            # Case 4: Multiplication by zero
            (
                torch.tensor([[1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
                torch.tensor([[0, 0, 0, 0]]),
                torch.zeros(1, 20),  # Expect all coefficients to be zero
            ),
        ],
    )
    def test_values(self, a_coeffs, b_coeffs, expected_coeffs, device, dtype):
        a = a_coeffs.to(device, dtype)
        b = b_coeffs.to(device, dtype)
        expected = expected_coeffs.to(device, dtype)
        result = solver.multiply_deg_two_one_poly(a, b)
        self.assert_close(result, expected, rtol=1e-4, atol=1e-4)


class TestDeterminantToPolynomial(BaseTester):
    def test_smoke(self, device, dtype):
        A = torch.rand(1, 3, 13, device=device, dtype=dtype)
        poly = solver.determinant_to_polynomial(A)
        assert poly.shape == (1, 11)

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        A = torch.rand(B, 3, 13, device=device, dtype=dtype)
        poly = solver.determinant_to_polynomial(A)
        assert poly.shape == (B, 11)

    @pytest.mark.parametrize(
        "A_in, expected_poly_coeffs",
        [
            # Case 1: An all-zero input should result in an all-zero polynomial.
            (
                torch.zeros(1, 3, 13),
                torch.zeros(1, 11),
            ),
            # Case 2: A sparse input designed to activate only the first term of cs[:, 10].
            # A[0,0,0]=2, A[0,1,4]=3, A[0,2,8]=5 -> term is 2*3*5 = 30.
            (
                torch.zeros(1, 3, 13).index_put(
                    (torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2]), torch.tensor([0, 4, 8])),
                    torch.tensor([2.0, 3.0, 5.0]),
                ),
                torch.zeros(1, 11).index_put((torch.tensor([0]), torch.tensor([10])), torch.tensor([30.0])),
            ),
            # Case 3: A sparse input designed to activate only one negative term in cs[:, 0].
            # A[0,0,7]=2, A[0,1,3]=3, A[0,2,12]=5 -> term is -A[0,7]*A[1,3]*A[2,12] = -30
            (
                torch.zeros(1, 3, 13).index_put(
                    (torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2]), torch.tensor([7, 3, 12])),
                    torch.tensor([2.0, 3.0, 5.0]),
                ),
                torch.zeros(1, 11).index_put((torch.tensor([0]), torch.tensor([0])), torch.tensor([-30.0])),
            ),
        ],
    )
    def test_values(self, A_in, expected_poly_coeffs, device, dtype):
        # Move tensor data to the target device and dtype
        A = A_in.to(device, dtype)
        expected = expected_poly_coeffs.to(device, dtype)

        # Compute the result
        result = solver.determinant_to_polynomial(A)

        # Compare result with expected values
        self.assert_close(result, expected, rtol=1e-5, atol=1e-5)


class TestQuarticSolver(BaseTester):
    def test_smoke(self, device, dtype):
        coeffs = torch.rand(1, 5, device=device, dtype=dtype)
        roots = solver.solve_quartic(coeffs)
        assert roots.shape == (1, 4)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 7])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        coeffs = torch.rand(B, 5, device=device, dtype=dtype)
        roots = solver.solve_quartic(coeffs)
        assert roots.shape == (B, 4)

    @pytest.mark.parametrize(
        "coeffs, expected_solutions",
        [
            # Case 1: Distinct Real Roots
            # x^4 - 10x^3 + 35x^2 - 50x + 24 = 0 -> Roots: 1, 2, 3, 4
            (
                torch.tensor([[1.0, -10.0, 35.0, -50.0, 24.0]]),
                torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            ),
            # Case 2: Biquadratic (Symmetric)
            # x^4 - 5x^2 + 4 = 0 -> Roots: 1, -1, 2, -2
            (
                torch.tensor([[1.0, 0.0, -5.0, 0.0, 4.0]]),
                torch.tensor([[-2.0, -1.0, 1.0, 2.0]]),
            ),
            # Case 3: Double Roots
            # (x-2)^2 * (x-3) * (x+1) -> Roots: -1, 2, 2, 3
            (
                torch.tensor([[1.0, -6.0, 9.0, 4.0, -12.0]]),
                torch.tensor([[-1.0, 2.0, 2.0, 3.0]]),
            ),
            # Case 4: Cubic Fallback (a=0)
            # 0x^4 + x^3 - 6x^2 + 11x - 6 = 0 -> Roots: 1, 2, 3. Last col 0.
            (
                torch.tensor([[0.0, 1.0, -6.0, 11.0, -6.0]]),
                torch.tensor([[1.0, 2.0, 3.0, 0.0]]),
            ),
            # Case 5: Degenerate / All Zeros
            # x^4 = 0 -> Roots: 0, 0, 0, 0
            (
                torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
            ),
            # Case 6: Complex Roots (Should be 0s per contract)
            # x^4 + 1 = 0 -> Roots: +/- sqrt(i) ... all complex -> 0, 0, 0, 0
            (
                torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]]),
                torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
            ),
            # Case 7: Mixed Real/Complex
            # x^4 - 1 = 0 -> Roots: 1, -1, i, -i -> Real: 1, -1. Others 0.
            (
                torch.tensor([[1.0, 0.0, 0.0, 0.0, -1.0]]),
                torch.tensor([[-1.0, 1.0, 0.0, 0.0]]),
            ),
        ],
    )
    def test_solve_quartic(self, coeffs, expected_solutions, device, dtype):
        coeffs = coeffs.to(device, dtype)
        expected_solutions = expected_solutions.to(device, dtype)

        roots = solver.solve_quartic(coeffs)

        # Sort roots to ensure order-invariant comparison
        # We sort both expected and actual to match this behavior.
        roots_sorted, _ = torch.sort(roots, dim=-1)
        expected_sorted, _ = torch.sort(expected_solutions, dim=-1)

        self.assert_close(roots_sorted, expected_sorted, rtol=1e-3, atol=1e-3)

    def test_random(self, device, dtype):
        # Generate random roots and construct coefficients to ensure valid solutions exist
        B = 10
        true_roots = torch.randn(B, 4, device=device, dtype=dtype)
        
        # Sort true roots for comparison later
        true_roots_sorted, _ = torch.sort(true_roots, dim=-1)
        
        r1, r2, r3, r4 = true_roots.unbind(-1)

        # Construct polynomial coefficients from roots
        # (x-r1)(x-r2)(x-r3)(x-r4) = 0
        a = torch.ones(B, device=device, dtype=dtype)
        b = -(r1 + r2 + r3 + r4)
        c = r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4
        d = -(r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4)
        e = r1 * r2 * r3 * r4

        coeffs = torch.stack([a, b, c, d, e], dim=-1)
        computed_roots = solver.solve_quartic(coeffs)
        
        computed_roots_sorted, _ = torch.sort(computed_roots, dim=-1)

        # 1. Check Residuals (Equation satisfaction)
        residuals = (
            coeffs[:, 0:1] * computed_roots**4
            + coeffs[:, 1:2] * computed_roots**3
            + coeffs[:, 2:3] * computed_roots**2
            + coeffs[:, 3:4] * computed_roots
            + coeffs[:, 4:5]
        )
        self.assert_close(residuals, torch.zeros_like(residuals), atol=1e-3, rtol=1e-3)

        # 2. Check Root Matching (Stronger Test)
        # Since we synthesized the coefficients from real roots, we expect 
        # to recover exactly those roots (no complex outputs).
        self.assert_close(computed_roots_sorted, true_roots_sorted, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        # Use a specific polynomial with distinct roots to ensure gradient stability
        # x^4 - 10x^3 + 35x^2 - 50x + 24 = 0
        # Avoid double roots for gradcheck as gradients are undefined/infinite there.
        coeffs = torch.tensor(
            [[1.0, -10.0, 35.0, -50.0, 24.0]],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        self.gradcheck(solver.solve_quartic, (coeffs,), raise_exception=True, fast_mode=True)
    def test_smoke(self, device, dtype):
        coeffs = torch.rand(1, 5, device=device, dtype=dtype)
        roots = solver.solve_quartic(coeffs)
        assert roots.shape == (1, 4)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 7])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        coeffs = torch.rand(B, 5, device=device, dtype=dtype)
        roots = solver.solve_quartic(coeffs)
        assert roots.shape == (B, 4)

    @pytest.mark.parametrize(
        "coeffs, expected_solutions",
        [
            # Case 1: x^4 - 10x^3 + 35x^2 - 50x + 24 = 0 -> Roots: 1, 2, 3, 4
            (
                torch.tensor([[1.0, -10.0, 35.0, -50.0, 24.0]]),
                torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            ),
            # Case 2: x^4 - 5x^2 + 4 = 0 -> Roots: 1, -1, 2, -2
            (
                torch.tensor([[1.0, 0.0, -5.0, 0.0, 4.0]]),
                torch.tensor([[-2.0, -1.0, 1.0, 2.0]]),
            ),
            # Case 3: (x-2)^2 * (x-3) * (x+1) -> Roots: -1, 2, 2, 3
            (
                torch.tensor([[1.0, -6.0, 9.0, 4.0, -12.0]]),
                torch.tensor([[-1.0, 2.0, 2.0, 3.0]]),
            ),
            # Case 4: Fallback to Cubic (a=0)
            # 0x^4 + x^3 - 6x^2 + 11x - 6 = 0 -> Roots: 1, 2, 3. Last col remains 0.
            (
                torch.tensor([[0.0, 1.0, -6.0, 11.0, -6.0]]),
                torch.tensor([[1.0, 2.0, 3.0, 0.0]]),
            ),
        ],
    )
    def test_solve_quartic(self, coeffs, expected_solutions, device, dtype):
        coeffs = coeffs.to(device, dtype)
        expected_solutions = expected_solutions.to(device, dtype)

        roots = solver.solve_quartic(coeffs)

        # Sort roots to ensure order-invariant comparison
        roots_sorted, _ = torch.sort(roots, dim=-1)
        expected_sorted, _ = torch.sort(expected_solutions, dim=-1)

        self.assert_close(roots_sorted, expected_sorted, rtol=1e-3, atol=1e-3)

    def test_random(self, device, dtype):
        # Generate random roots and construct coefficients to ensure valid solutions exist
        B = 5
        true_roots = torch.randn(B, 4, device=device, dtype=dtype)
        r1, r2, r3, r4 = true_roots.unbind(-1)

        a = torch.ones(B, device=device, dtype=dtype)
        b = -(r1 + r2 + r3 + r4)
        c = r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4
        d = -(r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4)
        e = r1 * r2 * r3 * r4

        coeffs = torch.stack([a, b, c, d, e], dim=-1)
        computed_roots = solver.solve_quartic(coeffs)

        residuals = (
            coeffs[:, 0:1] * computed_roots**4
            + coeffs[:, 1:2] * computed_roots**3
            + coeffs[:, 2:3] * computed_roots**2
            + coeffs[:, 3:4] * computed_roots
            + coeffs[:, 4:5]
        )

        self.assert_close(residuals, torch.zeros_like(residuals), atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        # Use a specific polynomial with distinct roots to ensure gradient stability
        # x^4 - 10x^3 + 35x^2 - 50x + 24 = 0
        coeffs = torch.tensor(
            [[1.0, -10.0, 35.0, -50.0, 24.0]],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        self.gradcheck(solver.solve_quartic, (coeffs,), raise_exception=True, fast_mode=True)
