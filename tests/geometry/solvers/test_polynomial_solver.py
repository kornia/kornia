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

import numpy as np
import pytest
import torch

import kornia.geometry.solvers as solver
from kornia.core.exceptions import ShapeError

from testing.base import BaseTester


def _monic_from_roots(roots: torch.Tensor) -> torch.Tensor:
    """Expand prod (x - r_i) row by row into monic coefficients, highest power first, in float64."""
    roots = roots.to(torch.float64)
    coeffs = torch.ones(roots.shape[0], 1, dtype=torch.float64)
    for i in range(roots.shape[1]):
        r = roots[:, i : i + 1]
        coeffs = torch.cat([coeffs, torch.zeros(roots.shape[0], 1, dtype=torch.float64)], dim=1)
        coeffs[:, 1:] = coeffs[:, 1:] - r * coeffs[:, :-1]
    return coeffs


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

    def test_gradcheck(self, device):
        # Deterministic, strictly positive discriminant (b^2 - 4ac = 1): `torch.rand` draws
        # all-positive coefficients, whose discriminant is usually negative, and the no-real-root
        # branch returns zeros with an identically zero gradient -- so a random draw checks
        # nothing roughly three times out of four. The zero-discriminant triple is excluded on
        # purpose: `sqrt` is not differentiable at 0.
        coeffs = torch.tensor([[1.0, -5.0, 6.0]], device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(solver.solve_quadratic, (coeffs,))


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
            (torch.tensor([[0.0, 0.0, 2.0, -6.0]]), torch.tensor([[3.0, 0.0, 0.0]])),  # handle first order
            (torch.tensor([[0.0, 1.0, -5.0, 6.0]]), torch.tensor([[3.0, 2.0, 0.0]])),  # handle second order
        ],
    )
    def test_solve_quadratic_in_cubic(self, coeffs, expected_solutions, device, dtype):
        roots = solver.solve_cubic(coeffs)
        self.assert_close(roots[0], expected_solutions[0], rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device):
        # Deterministic three-distinct-real-roots case (roots 2, -3, -1/2), so the check does not
        # depend on an unseeded draw. Repeated roots are excluded: they sit on the `sqrt` kink.
        coeffs = torch.tensor([[2.0, 3.0, -11.0, -6.0]], device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(solver.solve_cubic, (coeffs,))

    def test_convention_gradient_is_finite_at_the_acos_boundary_4290(self, device, dtype):
        # #4290: d(acos)/dx = -1/sqrt(1-x^2) is unbounded at x = +-1. solve_cubic's D<=0
        # branch computes acos(R / sqrt(-Q3)); the branch condition guarantees the ratio is in
        # [-1, 1], but a cubic with a repeated or near-repeated root pushes it to exactly that
        # boundary, where the VALUE is fine but the DERIVATIVE diverges -- same shape as the
        # acos/asin boundary in quaternion_exp_to_log/euler_from_quaternion (#4007, fixed in
        # #4228), a different call site not covered by that fix.
        #
        # This resolvent cubic comes from kornia's OWN existing double-root quartic fixture
        # (TestQuarticSolver.test_solve_quartic, "Case 3: Double Roots": (x-2)^2(x-3)(x+1),
        # coeffs [1, -6, 9, 4, -12]) -- already in the suite, already passes on forward value,
        # because that test never calls .backward(). It fails immediately if you do.
        coeffs = torch.tensor([[1.0, -6.0, 9.0, 4.0, -12.0]], device=device, dtype=dtype, requires_grad=True)
        roots = solver.solve_quartic(coeffs)
        roots.sum().backward()
        assert bool(torch.isfinite(coeffs.grad).all()), coeffs.grad

        # The forward value is unaffected by the gradient guard: unchanged, sorted match to the
        # documented roots -1, 2, 2, 3.
        expected = torch.tensor([[-1.0, 2.0, 2.0, 3.0]], device=device, dtype=dtype)
        roots_sorted, _ = torch.sort(roots.detach(), dim=-1)
        expected_sorted, _ = torch.sort(expected, dim=-1)
        self.assert_close(roots_sorted, expected_sorted, rtol=1e-3, atol=1e-3)

        # Second, independent repeated-root cubic exercising solve_cubic directly (not via a
        # quartic's resolvent): (x-1)^2(x-4) = x^3 - 6x^2 + 9x - 4. Verified this lands exactly
        # at the acos boundary (Q=-1, R=1, Q3=-1, D=Q3+R^2=0, ratio=R/sqrt(-Q3)=1.0 exactly) via
        # the D<=0 branch (Q != 0, so this is not the separate Q==0-and-R==0 triple-root path).
        cubic_coeffs = torch.tensor([[1.0, -6.0, 9.0, -4.0]], device=device, dtype=dtype, requires_grad=True)
        cubic_roots = solver.solve_cubic(cubic_coeffs)
        cubic_roots.sum().backward()
        assert bool(torch.isfinite(cubic_coeffs.grad).all()), cubic_coeffs.grad

    def test_convention_gradient_does_not_leak_across_batch_rows_4334(self, device, dtype):
        # #4334: the D > 0 branch keyed its work off `abs(R) > 1e-16` alone, so it evaluated
        # sqrt(D) on D < 0 rows too. Those rows are never read back, but `-Q / nan` stays in
        # the graph and DivBackward0 returns nan, which reaches every coefficient. The result
        # is that a row differentiates fine alone and gives nan in a mixed batch.
        three = [1.0, -7.0, 14.0, -8.0]  # (x-1)(x-2)(x-4): D < 0, three real roots, R != 0
        one = [1.0, 0.0, 1.0, -2.0]  # (x-1)(x^2+x+2): D > 0, one real root, Q != 0

        alone = torch.tensor([three], device=device, dtype=dtype, requires_grad=True)
        solver.solve_cubic(alone).sum().backward()

        mixed = torch.tensor([three, one], device=device, dtype=dtype, requires_grad=True)
        solver.solve_cubic(mixed).sum().backward()

        assert bool(torch.isfinite(mixed.grad).all()), mixed.grad
        # Batching must not change the answer either, not merely keep it finite.
        self.assert_close(mixed.grad[0], alone.grad[0])

    def test_convention_batched_forward_is_unchanged_by_neighbours_4334(self, device, dtype):
        # The forward pass was always correct; pin that, so a later fix that repairs the
        # gradient by perturbing the value is caught here.
        three = [1.0, -7.0, 14.0, -8.0]
        one = [1.0, 0.0, 1.0, -2.0]
        alone = torch.tensor([three], device=device, dtype=dtype)
        mixed = torch.tensor([three, one], device=device, dtype=dtype)
        self.assert_close(solver.solve_cubic(mixed)[0], solver.solve_cubic(alone)[0])

    @pytest.mark.parametrize(
        "coeffs, root",
        [
            ([1.0, 0.0, 0.0, 1.0], -1.0),  # x^3 + 1: Q == 0, R < 0
            ([1.0, 0.0, 0.0, 8.0], -2.0),  # x^3 + 8
            ([1.0, -7.5, 18.75, -15.5], 2.0),  # (x - 2)(x^2 - 5.5x + 7.75): Q == 0, R < 0 after the shift
            ([1.0, 0.0, 0.0, -1.0], 1.0),  # x^3 - 1: Q == 0, R > 0
        ],
    )
    def test_q_zero_real_cube_root_4832(self, coeffs, root, device, dtype):
        # #4832: the Q == 0 branch took torch.pow(2R, 1/3), which is nan for R < 0.
        x = torch.tensor([coeffs], device=device, dtype=dtype, requires_grad=True)
        roots = solver.solve_cubic(x)
        expected = torch.tensor([[root, 0.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(roots.detach(), expected)

        roots[:, 0].sum().backward()
        assert bool(torch.isfinite(x.grad).all()), x.grad
        if dtype in (torch.float32, torch.float64):
            # Implicit-function derivative of a simple root r of p: dr/dc_k = -r^(3 - k) / p'(r).
            # Q == 0 is an exact-equality branch, but the root still depends on Q (on c for x^3 + 1).
            a, b, c, _ = coeffs
            dp = 3 * a * root**2 + 2 * b * root + c
            expected_grad = torch.tensor([[-(root ** (3 - k)) / dp for k in range(4)]], device=device, dtype=dtype)
            self.assert_close(x.grad, expected_grad)

    @pytest.mark.parametrize("e", [3e-3, 1e-3, -3e-3, -1e-3])
    def test_odd_cubic_with_small_q_4856(self, e, device, dtype):
        # #4856: for x^3 - e*x, R == 0 and |Q| = |e| / 3 is small enough that Q^3 underflowed to 0 in
        # float16, so D == 0 and all three roots came back nan (e > 0 divided 0 by 0, e < 0 took
        # sqrt(-Q) of a positive Q).
        x = torch.tensor([[1.0, 0.0, -e, 0.0]], device=device, dtype=dtype, requires_grad=True)
        roots = solver.solve_cubic(x)
        s = e**0.5 if e > 0 else 0.0
        expected = torch.tensor([[s, -s, 0.0]], device=device, dtype=dtype)
        self.assert_close(roots.detach(), expected)

        roots.sum().backward()
        assert bool(torch.isfinite(x.grad).all()), x.grad


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
            # Case 8: Resolvent cubic with Q == 0 and R < 0 (#4832)
            # x^4 - 3x^2 - 0.75 = 0 -> Real: +/- sqrt((3 + sqrt(12)) / 2). Others 0.
            # Its resolvent y^3 + 3y^2 + 3y + 9 has Q == 0 and R = -4 < 0, which used to make every root nan.
            (
                torch.tensor([[1.0, 0.0, -3.0, 0.0, -0.75]]),
                torch.tensor([[-1.7977905, 1.7977905, 0.0, 0.0]]),
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

    @pytest.mark.parametrize(
        "coeffs, expected_solutions",
        [
            (
                [1.0, 2.0, 0.0, 0.0, -16.0],
                [-2.760555, 0.0, 0.0, 1.638345],
            ),
            (
                [1.0, 0.0, -4.0, 0.0, -5.0],
                [-(5.0**0.5), 0.0, 0.0, 5.0**0.5],
            ),
        ],
    )
    def test_real_roots_with_complex_pair(self, coeffs, expected_solutions, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("This regression is limited to float32 and float64.")

        coeffs_tensor = torch.tensor([coeffs], device=device, dtype=dtype)
        expected = torch.tensor([expected_solutions], device=device, dtype=dtype)

        roots = solver.solve_quartic(coeffs_tensor)
        roots_sorted, _ = torch.sort(roots, dim=-1)

        self.assert_close(roots_sorted, expected, rtol=1e-4, atol=1e-4)

    def test_resolvent_filter_half_precision(self, device, dtype):
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("This regression targets half-precision resolvent validation.")

        coeffs = torch.tensor([[1.0, -10.0, 35.0, -50.0, 24.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device, dtype=dtype)

        roots = solver.solve_quartic(coeffs)
        assert bool(torch.isfinite(roots).all()), roots
        assert bool((roots != 0).all()), roots
        roots_sorted, _ = torch.sort(roots, dim=-1)

        self.assert_close(roots_sorted, expected)

    @pytest.mark.parametrize(
        "coeffs",
        [
            [1.0, 0.0, 0.0, 1e-6, -16.0],
            [1.0, 1e-6, 0.0, 0.0, -16.0],
        ],
    )
    def test_near_biquadratic_avoids_R_division(self, coeffs, device, dtype):
        if dtype != torch.float64:
            pytest.skip("This regression targets the float64 R^2 tolerance.")

        coeffs_tensor = torch.tensor([coeffs], device=device, dtype=dtype)
        expected = torch.tensor([[-2.0, 0.0, 0.0, 2.0]], device=device, dtype=dtype)

        roots = solver.solve_quartic(coeffs_tensor)
        roots_sorted, _ = torch.sort(roots, dim=-1)

        self.assert_close(roots_sorted, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize(
        "coeffs, expected_solutions",
        [
            (
                [1.0, -0.046868806472256, 0.0, 0.0, -3.9079498911375055],
                [-1.3944338896982997, 0.0, 0.0, 1.417871547980907],
            ),
            (
                [1.0, 0.0, 0.876698529368765, 1e-6, -0.007128260662146779],
                [-0.08976001409129965, 0.0, 0.0, 0.08975889403473052],
            ),
        ],
    )
    def test_small_R_sq_constant_term_identity(self, coeffs, expected_solutions, device, dtype):
        if dtype != torch.float64:
            pytest.skip("This regression targets float64 resolvent conditioning.")

        coeffs_tensor = torch.tensor([coeffs], device=device, dtype=dtype)
        expected = torch.tensor([expected_solutions], device=device, dtype=dtype)

        roots = solver.solve_quartic(coeffs_tensor)
        roots_sorted, _ = torch.sort(roots, dim=-1)

        self.assert_close(roots_sorted, expected, rtol=0.0, atol=1e-5)

    def test_direct_coefficients_against_numpy(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("NumPy reference coverage is limited to float32 and float64.")

        rng = np.random.default_rng(4346)
        coeffs_np = np.concatenate([np.ones((64, 1)), rng.uniform(-5.0, 5.0, size=(64, 4))], axis=1)
        expected_np = np.zeros((64, 4))

        for idx, row in enumerate(coeffs_np):
            roots = np.roots(row)
            real_roots = roots.real[np.abs(roots.imag) <= 1e-7]
            expected_np[idx, : len(real_roots)] = real_roots

        coeffs_tensor = torch.tensor(coeffs_np, device=device, dtype=dtype)
        expected = torch.tensor(expected_np, device=device, dtype=dtype)

        computed_roots = solver.solve_quartic(coeffs_tensor)
        computed_roots_sorted, _ = torch.sort(computed_roots, dim=-1)
        expected_sorted, _ = torch.sort(expected, dim=-1)

        if dtype == torch.float64:
            self.assert_close(computed_roots_sorted, expected_sorted, rtol=0.0, atol=1e-5)
        else:
            self.assert_close(computed_roots_sorted, expected_sorted, rtol=1e-3, atol=1e-3)

    def test_constant_term_E_float32_regression(self, device, dtype):
        if dtype != torch.float32:
            pytest.skip("This regression targets the float32 near-zero-R conditioning failure.")

        coeffs = torch.tensor([[1.0, -2.4491360, 1.1192101, -2.0354464, -3.8231988]], device=device, dtype=dtype)
        expected_real = torch.tensor([-0.782185, 2.552848], device=device, dtype=dtype)

        roots = solver.solve_quartic(coeffs)
        assert bool(torch.isfinite(roots).all()), roots
        assert int(torch.count_nonzero(roots)) == 2, roots

        real_roots = torch.sort(roots[roots != 0]).values
        self.assert_close(real_roots, expected_real, rtol=1e-3, atol=1e-3)

        residuals = (
            coeffs[0, 0] * real_roots**4
            + coeffs[0, 1] * real_roots**3
            + coeffs[0, 2] * real_roots**2
            + coeffs[0, 3] * real_roots
            + coeffs[0, 4]
        )
        self.assert_close(residuals, torch.zeros_like(residuals), rtol=0.0, atol=1e-3)

    def test_resolvent_fallback_float32_regression(self, device, dtype):
        if dtype != torch.float32:
            pytest.skip("This regression targets float32 resolvent fallback selection.")

        coeffs = torch.tensor(
            [[1.0, 4.340823173522949, 3.653407096862793, 3.0506274700164795, -2.266538143157959]],
            device=device,
            dtype=dtype,
        )
        expected_real = torch.tensor([-3.61119394, 0.41863132], device=device, dtype=dtype)

        roots = solver.solve_quartic(coeffs)
        assert bool(torch.isfinite(roots).all()), roots
        assert int(torch.count_nonzero(roots)) == 2, roots

        real_roots = torch.sort(roots[roots != 0]).values
        self.assert_close(real_roots, expected_real, rtol=1e-3, atol=1e-3)

        residuals = (
            coeffs[0, 0] * real_roots**4
            + coeffs[0, 1] * real_roots**3
            + coeffs[0, 2] * real_roots**2
            + coeffs[0, 3] * real_roots
            + coeffs[0, 4]
        )
        self.assert_close(residuals, torch.zeros_like(residuals), rtol=0.0, atol=1e-3)

    def test_E_reconstruction_precision_float64(self, device, dtype):
        if dtype != torch.float64:
            pytest.skip("This regression targets float64 Ferrari factorization precision.")

        # (x^2 + 3x + 2)(x^2 - 5x + 2.000001)
        coeffs = torch.tensor([[1.0, -2.0, -10.999999, -3.999997, 4.000002]], device=device, dtype=dtype)
        discriminant = torch.sqrt(torch.tensor(16.999996, device=device, dtype=dtype))
        expected = torch.tensor(
            [[-2.0, -1.0, (5.0 - discriminant) / 2.0, (5.0 + discriminant) / 2.0]],
            device=device,
            dtype=dtype,
        )

        roots = torch.sort(solver.solve_quartic(coeffs), dim=-1).values
        expected = torch.sort(expected, dim=-1).values
        self.assert_close(roots, expected, rtol=0.0, atol=1e-12)

        residuals = (
            coeffs[:, 0:1] * roots**4
            + coeffs[:, 1:2] * roots**3
            + coeffs[:, 2:3] * roots**2
            + coeffs[:, 3:4] * roots
            + coeffs[:, 4:5]
        )
        self.assert_close(residuals, torch.zeros_like(residuals), rtol=0.0, atol=1e-12)

    def test_biquadratic_R_sq_relative_snap(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("This regression targets full-precision R^2 cancellation handling.")

        coeffs = torch.tensor([[1.0, 0.0, 2.9696312, 0.0, -2.3985832]], device=device, dtype=dtype)
        expected_real = torch.tensor([-0.8128378975367444, 0.8128378975367438], device=device, dtype=dtype)

        roots = solver.solve_quartic(coeffs)
        assert bool(torch.isfinite(roots).all()), roots
        assert int(torch.count_nonzero(roots)) == 2, roots

        real_roots = torch.sort(roots[roots != 0]).values
        self.assert_close(real_roots, expected_real, rtol=0.0, atol=5e-6)

        residuals = (
            coeffs[0, 0] * real_roots**4
            + coeffs[0, 1] * real_roots**3
            + coeffs[0, 2] * real_roots**2
            + coeffs[0, 3] * real_roots
            + coeffs[0, 4]
        )
        self.assert_close(residuals, torch.zeros_like(residuals), rtol=0.0, atol=2e-5)

    def test_biquadratic_R_sq_relative_snap_float64_issue_literal(self, device, dtype):
        if dtype != torch.float64:
            pytest.skip("This regression targets the float64 R^2 cancellation budget.")

        coeffs = torch.tensor([[1.0, 0.0, -4.0, 0.0, -5.0]], device=device, dtype=dtype)
        expected_real = torch.tensor([-(5.0**0.5), 5.0**0.5], device=device, dtype=dtype)

        roots = solver.solve_quartic(coeffs)
        assert bool(torch.isfinite(roots).all()), roots
        assert int(torch.count_nonzero(roots)) == 2, roots

        real_roots = torch.sort(roots[roots != 0]).values
        self.assert_close(real_roots, expected_real, rtol=1e-12, atol=1e-12)

        residuals = (
            coeffs[0, 0] * real_roots**4
            + coeffs[0, 1] * real_roots**3
            + coeffs[0, 2] * real_roots**2
            + coeffs[0, 3] * real_roots
            + coeffs[0, 4]
        )
        self.assert_close(residuals, torch.zeros_like(residuals), rtol=0.0, atol=1e-12)

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

    @pytest.mark.parametrize(
        ("coeffs", "expected", "expected_grad"),
        [
            # x^4 - 16 = (x^2-4)(x^2+4): two real roots, +-2. R_sq lands exactly on 0.
            ([1.0, 0.0, 0.0, 0.0, -16.0], [2.0, -2.0], [0.0, -0.5, 0.0, 0.0, 0.0]),
            # x^4 - 1 = (x^2-1)(x^2+1): two real roots, +-1. R_sq lands exactly on 0.
            ([1.0, 0.0, 0.0, 0.0, -1.0], [1.0, -1.0], [0.0, -0.5, 0.0, 0.0, 0.0]),
            # (x^2+1)(x^2+4): no real roots. R_sq < 0 makes R zero, while the constant-term
            # identity for E also has radicand 0 exactly -- the second sqrt site.
            ([1.0, 0.0, 5.0, 0.0, 4.0], None, [0.0, 0.0, 0.0, 0.0, 0.0]),
        ],
    )
    def test_convention_gradient_is_finite_for_a_pure_biquadratic_4229(
        self, coeffs, expected, expected_grad, device, dtype
    ):
        # A quartic with no x^3 and no x^2 term puts R_sq exactly on 0, and
        # `torch.clamp(R_sq, min=0.0).sqrt()` does not guard that: d(sqrt)/dx is unbounded at 0,
        # and on torch < 2.14 clamp passes the incoming gradient through at the bound rather
        # than zeroing it (#4229). kornia supports torch>=2.5.1, so the guard was a no-op on the
        # older half of the supported range and the backward returned nan. On torch >= 2.14 clamp
        # already zeroes the boundary gradient, so these pins pass on base on those legs; the
        # 2.5.1 and 2.9.1 CI legs carry the discrimination.
        c = torch.tensor([coeffs], device=device, dtype=dtype, requires_grad=True)
        roots = solver.solve_quartic(c)
        roots.sum().backward()

        assert bool(torch.isfinite(c.grad).all()), c.grad
        # Pin the value the guard produces, not just its finiteness: a "detach everything"
        # pseudo-fix keeps the gradient finite but does not reproduce these numbers. This is a
        # convention at the zero-radicand sqrt boundaries (#4229/#4339), not a claim about the
        # mathematical quartic-root Jacobian.
        self.assert_close(
            c.grad[0],
            torch.tensor(expected_grad, device=device, dtype=dtype),
            rtol=1e-4,
            atol=1e-4,
        )
        # The forward pass was always correct; pin it, so a fix that repairs the gradient by
        # moving the value is caught here.
        if expected is None:
            # No real roots: every entry is the zero placeholder.
            self.assert_close(
                roots.detach()[0],
                torch.zeros(4, device=device, dtype=dtype),
                rtol=1e-4,
                atol=1e-4,
            )
        else:
            real = torch.sort(roots.detach()[0][:2]).values
            self.assert_close(
                real,
                torch.sort(torch.tensor(expected, device=device, dtype=dtype)).values,
                rtol=1e-4,
                atol=1e-4,
            )

    def test_convention_gradient_does_not_leak_across_batch_rows_4334(self, device, dtype):
        # #4334 through the resolvent cubic: a four-real-root quartic batched with a
        # two-real-root one took nan gradients from solve_cubic's D > 0 branch.
        four = [1.0, -10.0, 35.0, -50.0, 24.0]  # (x-1)(x-2)(x-3)(x-4)
        two = [1.0, 0.0, 0.0, 0.0, -16.0]  # x^4 - 16

        alone = torch.tensor([four], device=device, dtype=dtype, requires_grad=True)
        solver.solve_quartic(alone).sum().backward()

        mixed = torch.tensor([four, two], device=device, dtype=dtype, requires_grad=True)
        solver.solve_quartic(mixed).sum().backward()

        # Check row 0 for the cross-batch contamination from #4334. Row 1's separate
        # zero-radicand gradient convention was fixed in #4339 and is covered above.
        assert bool(torch.isfinite(mixed.grad[0]).all()), mixed.grad
        self.assert_close(mixed.grad[0], alone.grad[0])

    def test_no_spurious_real_roots_for_near_square_quartic_4474(self, device, dtype):
        # (x^2 + 2x + 5)(x^2 + 2x + 5.01) has no real roots, but its resolvent cubic has a
        # near-double root that float32 solve_cubic loses. Ferrari then fell back to R = 0,
        # both quadratics collapsed to x^2 + (A/2)x + y/2, and its roots came back as the
        # quartic's while leaving a residual of 25 (#4474).
        coeffs = torch.tensor([[1.0, 4.0, 14.01, 20.02, 25.05]], device=device, dtype=dtype)
        roots = solver.solve_quartic(coeffs)

        # Every returned value must satisfy the quartic it was given. Zeros are the
        # placeholder for "no real root" and are skipped, which is the correct answer here.
        for root in roots[0]:
            if root == 0.0:
                continue
            residual = (((root + 4.0) * root + 14.01) * root + 20.02) * root + 25.05
            assert bool(torch.abs(residual) < 1e-2), f"{root} is not a root, residual {residual}"

    def test_near_square_family_returns_no_real_roots_4474(self, device, dtype):
        # The same failure across the family rather than one literal, so a future change that
        # reintroduces it on neighbouring coefficients is caught too. Each row is
        # (x^2 + a x + b)(x^2 + a x + b + eps) with a discriminant that admits no real root.
        rows, a = [], 2.0
        for b in (5.0, 6.5, 8.0):
            for eps in (1e-3, 1e-2, 1e-1):
                c1 = b + eps
                # expand (x^2 + a x + b)(x^2 + a x + c1)
                rows.append([1.0, 2 * a, b + c1 + a * a, a * (b + c1), b * c1])
        coeffs = torch.tensor(rows, device=device, dtype=dtype)
        roots = solver.solve_quartic(coeffs)

        assert bool((roots == 0.0).all()), f"expected the no-real-root placeholder, got {roots}"

    def test_a_genuine_root_ferrari_left_off_is_polished_not_dropped_4474(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("Root accuracy assertions are limited to float32 and float64.")
        # Four distinct real roots, nothing pathological. float32 Ferrari returns -0.02384 for the
        # root at -0.023854, a scaled residual of 1.1e-4, and a fixed 1e-4 cutoff replaced it with
        # the no-root placeholder (review of #4669). Polishing brings it onto the root instead.
        coeffs = torch.tensor(
            [[1.0, -0.3775281906, -71.65801239, -11.11165237, -0.2242867798]], device=device, dtype=dtype
        )
        expected = torch.tensor([[-8.19822634, -0.13136028, -0.02385378, 8.7309686]], device=device, dtype=dtype)
        roots = torch.sort(solver.solve_quartic(coeffs), dim=-1).values
        self.assert_close(roots, expected, rtol=1e-4, atol=1e-6)

    def test_separated_real_roots_are_never_dropped_4474(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("Root accuracy assertions are limited to float32 and float64.")
        # 2000 quartics with four real roots in [-10, 10] at least 0.5 apart: every root must come
        # back within the dtype's reach of its true value, and none as the zero placeholder.
        gen = torch.Generator().manual_seed(4474)
        gaps = torch.rand(2000, 3, generator=gen, dtype=torch.float64) * 4 + 0.5
        first = torch.rand(2000, 1, generator=gen, dtype=torch.float64) * (20 - gaps.sum(1, keepdim=True)) - 10
        true_roots = torch.cat([first, first + gaps.cumsum(1)], dim=1)
        coeffs = _monic_from_roots(true_roots)
        roots = torch.sort(solver.solve_quartic(coeffs.to(device=device, dtype=dtype)), dim=-1).values

        assert bool((roots != 0).all()), f"{int((roots == 0).sum())} roots replaced by the placeholder"
        self.assert_close(roots, true_roots.to(device=device, dtype=dtype), rtol=1e-3, atol=1e-3)

    def test_polish_does_not_repeat_a_simple_root_4474(self, device, dtype):
        # Two real roots and a complex pair. The quadratic that should hold the pair collapses to a
        # double candidate near the small real root, and polishing alone carried both copies onto
        # it: 0.0816 came back three times. A simple root is reported once.
        coeffs = torch.tensor([[1.0, 0.34461, -0.46098, 2.76681, -0.22301]], device=device, dtype=dtype)
        roots = solver.solve_quartic(coeffs)
        nonzero = roots[roots != 0]
        assert nonzero.numel() == 2, f"expected the two real roots and two placeholders, got {roots}"
        expected = torch.tensor([-1.666164, 0.081621], device=device, dtype=dtype)
        self.assert_close(torch.sort(nonzero).values, expected, rtol=1e-3, atol=1e-4)

    def test_close_distinct_simple_roots_are_both_kept_4474(self, device, dtype):
        # Roots 0.01, 0.0109, 10, 20 (review of #4669): a fixed coincidence window took the two small
        # roots for one and replaced 0.01 with the placeholder. The window is now each candidate's
        # own error bound, which two distinct roots do not share however close they are.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("Root accuracy assertions are limited to float32 and float64.")
        coeffs = torch.tensor([[1.0, -30.0209, 200.627109, -4.18327, 0.0218]], device=device, dtype=dtype)
        roots = torch.sort(solver.solve_quartic(coeffs), dim=-1).values
        expected = torch.tensor([[0.01, 0.0109, 10.0, 20.0]], device=device, dtype=dtype)
        # float32 cannot separate the pair better than ~1%; float64 places both exactly.
        tol = 2e-2 if dtype == torch.float32 else 1e-6
        self.assert_close(roots, expected, rtol=tol, atol=0.0)

    def test_double_root_is_still_reported_twice_4474(self, device, dtype):
        # (x - 2)^2 (x + 1)(x + 3) and (x^2 - 1)^2: the repeat rule must leave a genuine double
        # root alone, whether it comes from one quadratic or one copy from each.
        coeffs = torch.tensor([[1.0, 0.0, -9.0, 4.0, 12.0], [1.0, 0.0, -2.0, 0.0, 1.0]], device=device, dtype=dtype)
        roots = torch.sort(solver.solve_quartic(coeffs), dim=-1).values
        expected = torch.tensor([[-3.0, -1.0, 2.0, 2.0], [-1.0, -1.0, 1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(roots, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize(
        "coeffs, true_roots, dtypes",
        [
            # (x + 8.75)(x + 8.74)(x^2 - 2x + 17), and the same with roots -8.75 and -8.748. Polishing a
            # complex-pair placeholder from zero stopped a tenth away from the close pair, inside the
            # residual tolerance: -8.8777 twice, or -8.8840 four times.
            ([1.0, 15.49, 58.495, 144.38, 1300.075], [-8.75, -8.74], (torch.float32, torch.float64)),
            ([1.0, 15.498, 58.549, 144.376, 1301.265], [-8.75, -8.748], (torch.float32, torch.float64)),
            # A recovered placeholder must have converged: accepting it at any simple root returns -1.10588
            # for the near-double root at -1.1189.
            (
                [1.0, -2.4284734, -3.84090791, 6.128004724, 6.696065824],
                [-1.11887, -1.11886, 2.02579, 2.64041],
                (torch.float32, torch.float64),
            ),
            # ...relative to |x| itself: a bound of sqrt(eps) * max(1, |x|) is absolute below 1, and
            # returns 0.0008457 for the root at 0.00112, 25% off.
            (
                [1.0, -6.807518769, -104.0549685, 0.2346873751, -0.0001323169874],
                [-7.351354433, 0.001122449359, 0.001132718443, 14.15661803],
                (torch.float32,),
            ),
        ],
    )
    def test_returned_values_are_roots_4474(self, coeffs, true_roots, dtypes, device, dtype):
        if dtype not in dtypes:
            pytest.skip("This case pins behaviour in another dtype.")
        roots = solver.solve_quartic(torch.tensor([coeffs], device=device, dtype=dtype))[0]
        want = torch.tensor(true_roots, device=device, dtype=dtype)
        # Relative to the root itself, with no floor at 1, so a value near zero is held to the same standard.
        rtol = 1e-2 if dtype == torch.float32 else 1e-6
        for value in roots[roots != 0]:
            assert bool(((want - value).abs() <= rtol * want.abs()).any()), f"{value} is not a root: {roots.tolist()}"

    @pytest.mark.parametrize(
        "coeffs, expected, dtypes",
        [
            # Each case pins one rule of the polish/filter/dedupe step: changing that rule makes it fail.
            # Placeholders are judged apart from Ferrari's candidates: without it -8.8777 is returned twice.
            ([1.0, 15.49, 58.495, 144.38, 1300.075], [-8.75, -8.74], (torch.float32, torch.float64)),
            # Placeholders are polished: without it the root at 0.003 is lost next to roots of 1e2 to 1e3.
            (
                [1.0, -1088.503, -128841.7345, 329286.535, -986.7],
                [-110.0, 0.003, 2.5, 1196.0],
                (torch.float32, torch.float64),
            ),
            # The error bound comes only from simple roots: counting the double root drops 1.68.
            (
                _monic_from_roots(torch.tensor([[2.3, 2.3, -4.19, 1.68]], dtype=torch.float64))[0].tolist(),
                [-4.19, 1.68, 2.3, 2.3],
                (torch.float64,),
            ),
            # Coincidence factor 4, not 1: at 1 a second -1.5 survives (roots -2, -1.5, 4 +- 0.5i).
            ([1.0, -4.5, -8.75, 32.875, 48.75], [-2.0, -1.5], (torch.float32, torch.float64)),
            # Simple-root threshold 1e-2, not 1e-1: at 1e-1 -4031 is returned twice (roots -4031, -4689,
            # 231 +- 875i). The smaller case (roots -9, -5, 1 +- 3i) guards the same repeat at -9.
            (
                [1.0, 8258.0, 15691705.0, -1590869938.0, 15479948400000.0],
                [-4689.0, -4031.0],
                (torch.float32, torch.float64),
            ),
            ([1.0, 12.0, 27.0, 50.0, 450.0], [-9.0, -5.0], (torch.float32, torch.float64)),
            # ...and not 1e-3: at 1e-3 the double root at -2 loses a copy (roots -5, -2, -2, 2.25).
            ([1.0, 6.75, 3.75, -34.0, -45.0], [-5.0, -2.0, -2.0, 2.25], (torch.float32, torch.float64)),
            # The ulp floor in the coincidence window: without it 4.75 comes back twice (roots 4.75, -5, 9 +- 3i).
            ([1.0, -17.75, 61.75, 450.0, -2137.5], [-5.0, 4.75], (torch.float32, torch.float64)),
            # The residual tolerance, pinned from both sides. At sqrt(eps) instead of sqrt(eps) / 4, this
            # quartic with two complex pairs returns -7.2066 and -6.7561 twice each...
            ([1.0, 27.91975997, 297.7351036, 1435.935501, 2645.836994], [], (torch.float32,)),
            # ...and at sqrt(eps) / 16 both copies of the double root at 0.558935 are dropped.
            (
                [1.0, -7.636022673, 0.9114557167, 5.439310611, -2.089195035],
                [-0.901329, 0.558935, 0.558935, 7.419483],
                (torch.float32,),
            ),
            # A recovered placeholder's step bound, from below: at eps * |x| instead of sqrt(eps) * |x| the
            # root at 0.0009864 next to roots of 1 to 946 is lost.
            (
                [1.0, -1333.67141, 366869.5534, -369281.897, 363.9159878],
                [0.0009864, 1.009293, 386.1998282, 946.4613022],
                (torch.float32,),
            ),
        ],
    )
    def test_root_set_4474(self, coeffs, expected, dtypes, device, dtype):
        if dtype not in dtypes:
            pytest.skip("This case pins behaviour in another dtype.")
        roots = solver.solve_quartic(torch.tensor([coeffs], device=device, dtype=dtype))[0]
        found = torch.sort(roots[roots != 0]).values
        assert found.numel() == len(expected), f"expected {expected}, got {roots.tolist()}"
        # A double root in float32 lands ~sqrt(eps) apart; everything else here is far tighter.
        tol = 1e-2 if dtype == torch.float32 else 1e-6
        want = torch.tensor(sorted(expected), device=device, dtype=dtype)
        self.assert_close(found, want, rtol=tol, atol=tol)

    def test_ferrari_candidate_is_kept_over_a_recovered_copy_4474(self, device, dtype):
        # A placeholder recovered to 2.288697 and Ferrari's own 2.289372 are copies of the root at
        # 2.289375. The recovered one had two steps from zero and is 3e-4 off; keeping it by slot order
        # threw away the accurate one. Ferrari's candidate is kept.
        if dtype != torch.float32:
            pytest.skip("The recovered copy arises in float32.")
        coeffs = torch.tensor([[1.0, -7.111530047, 11.09636462, 16.30030766, -37.6143968]], device=device, dtype=dtype)
        roots = solver.solve_quartic(coeffs)[0]
        near = roots[(roots - 2.289375).abs() <= 1e-2 * 2.289375]
        assert near.numel() == 1, f"expected one copy of 2.289375, got {roots.tolist()}"
        assert abs(float(near) - 2.289375) <= 1e-5 * 2.289375, f"kept the less accurate copy: {float(near)}"

    def test_four_real_roots_survive_the_residual_filter_4474(self, device, dtype):
        # The guard rejects non-roots; it must not reject roots. A well-separated
        # four-real-root quartic and a biquadratic both keep every root.
        quartics = [
            [1.0, -10.0, 35.0, -50.0, 24.0],  # (x-1)(x-2)(x-3)(x-4)
            [1.0, 0.0, -5.0, 0.0, 4.0],  # (x^2-1)(x^2-4)
        ]
        expected = [[1.0, 2.0, 3.0, 4.0], [-2.0, -1.0, 1.0, 2.0]]
        roots = solver.solve_quartic(torch.tensor(quartics, device=device, dtype=dtype))

        for row, want in zip(roots, expected):
            got = torch.sort(row).values
            self.assert_close(got, torch.tensor(want, device=device, dtype=dtype).sort().values, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("leading", [0.0, 1e-9, 5e-7])
    def test_cubic_fallback_for_near_zero_leading_coefficient(self, leading, device, dtype):
        # (x - 1)(x - 2)(x - 3) behind a leading coefficient below the 1e-6 cubic-fallback tolerance
        # that float32 and the half-precision inputs solved in float32 share (#4498). float16 rounded
        # float64's 1e-12 to 0 and returned nan, and bfloat16 lost all three roots at 1e-9.
        if leading != 0.0 and dtype == torch.float64:
            pytest.skip("float64 keeps its 1e-12 tolerance and solves these rows as quartics")
        coeffs = torch.tensor([[leading, 1.0, -6.0, 11.0, -6.0]], device=device, dtype=dtype)

        roots = solver.solve_quartic(coeffs)

        expected = torch.cat(
            [solver.solve_cubic(coeffs[:, 1:]), torch.zeros((1, 1), device=device, dtype=dtype)], dim=-1
        )
        self.assert_close(roots, expected, rtol=0.0, atol=0.0)

    def test_leading_coefficient_above_fallback_tolerance_is_solved_as_quartic(self, device, dtype):
        # Just above the 1e-6 tolerance the row keeps its fourth root, near -1 / a - 6 = -500006
        # (beyond float16's range, so -inf there), instead of the cubic fallback's 0.
        coeffs = torch.tensor([[2e-6, 1.0, -6.0, 11.0, -6.0]], device=device, dtype=dtype)

        roots = solver.solve_quartic(coeffs)

        assert roots.min().item() < -1e5, roots


# determinant_to_polynomial input: three rows of 13 coefficients, each two cubics (columns 0-3, 4-7) and a quartic
# (columns 8-12), highest degree first; generated by [[((7 * k + 3 * r) % 11 - 5) / 4 for k in range(13)] for r in
# range(3)], so no row, column or coefficient block is symmetric.
_DET_ROWS = [
    [-1.25, 0.5, -0.5, 1.25, 0.25, -0.75, 1.0, 0.0, -1.0, 0.75, -0.25, -1.25, 0.5],
    [-0.5, 1.25, 0.25, -0.75, 1.0, 0.0, -1.0, 0.75, -0.25, -1.25, 0.5, -0.5, 1.25],
    [0.25, -0.75, 1.0, 0.0, -1.0, 0.75, -0.25, -1.25, 0.5, -0.5, 1.25, 0.25, -0.75],
]


def _polyval(coeffs: list, z: float) -> float:
    """Evaluate coefficients given highest degree first."""
    out = 0.0
    for c in coeffs:
        out = out * z + c
    return out


class TestConventionPolynomialSolvers(BaseTester):
    @pytest.mark.parametrize(
        "coeffs, expected",
        [
            ([1.0, -3.0, 2.0], [2.0, 1.0]),  # two real roots
            ([1.0, 0.0, 1.0], [0.0, 0.0]),  # x^2 + 1: no real root, both slots are 0.0
            ([1.0, -3.0, 0.0], [3.0, 0.0]),  # x^2 - 3x: a genuine root at 0 looks the same as a padded slot
        ],
    )
    def test_convention_solve_quadratic_real_roots_zero_padded(self, coeffs, expected, device, dtype):
        # Only real roots are returned; a missing real root is reported as 0.0 (a design choice, not NaN).
        out = solver.solve_quadratic(torch.tensor([coeffs], device=device, dtype=dtype))
        self.assert_close(out, torch.tensor([expected], device=device, dtype=dtype))

    @pytest.mark.parametrize(
        "fn, coeffs, roots",
        [
            # 2x^2 - 5x - 3 = (2x + 1)(x - 3)
            (solver.solve_quadratic, [2.0, -5.0, -3.0], [-0.5, 3.0]),
            # -2 (x - 3)(x + 0.5)(x - 1.25)
            (solver.solve_cubic, [-2.0, 7.5, -3.25, -3.75], [-0.5, 1.25, 3.0]),
            # 3 (x - 4)(x + 2)(x - 1)(x + 0.5)
            (solver.solve_quartic, [3.0, -7.5, -22.5, 15.0, 12.0], [-2.0, -0.5, 1.0, 4.0]),
        ],
        ids=["quadratic", "cubic", "quartic"],
    )
    def test_convention_solver_coefficient_layout_highest_degree_first(self, fn, coeffs, roots, device, dtype):
        # coeffs[0] multiplies the highest power, as in numpy.roots; read lowest degree first, the same rows have
        # the reciprocal roots (for instance 1/3 and -2 for the quadratic). Rows are batched (B, k + 1).
        out = fn(torch.tensor([coeffs], device=device, dtype=dtype))
        self.assert_close(out.sort(dim=-1).values, torch.tensor([roots], device=device, dtype=dtype))
        with pytest.raises(ShapeError):
            fn(torch.tensor(coeffs, device=device, dtype=dtype))

    def test_convention_solver_root_multiplicity_and_order(self, device, dtype):
        def solve(fn, coeffs):
            return fn(torch.tensor([coeffs], device=device, dtype=dtype))

        def expect(values):
            return torch.tensor([values], device=device, dtype=dtype)

        # solve_quadratic returns [(-b + sqrt(D)) / (2a), (-b - sqrt(D)) / (2a)]: the larger root first for a > 0,
        # the smaller first for a < 0 (roots 3 and -0.5 both times).
        self.assert_close(solve(solver.solve_quadratic, [1.0, -2.5, -1.5]), expect([3.0, -0.5]))
        self.assert_close(solve(solver.solve_quadratic, [-1.0, 2.5, 1.5]), expect([-0.5, 3.0]))
        # A repeated root is repeated, not padded.
        self.assert_close(solve(solver.solve_quadratic, [1.0, -6.0, 9.0]), expect([3.0, 3.0]))
        cubic = solve(solver.solve_cubic, [1.0, -3.0, 0.0, 4.0])  # (x - 2)^2 (x + 1)
        self.assert_close(cubic.sort(dim=-1).values, expect([-1.0, 2.0, 2.0]))
        # A single real root is in slot 0 of solve_cubic, followed by the zero padding: (x - 2)(x^2 + 1).
        self.assert_close(solve(solver.solve_cubic, [1.0, -2.0, 1.0, -2.0]), expect([2.0, 0.0, 0.0]))

    def test_convention_determinant_to_polynomial_output_lowest_degree_first(self, device, dtype):
        cs = solver.determinant_to_polynomial(torch.tensor([_DET_ROWS], device=device, dtype=dtype))
        assert cs.shape == (1, 11)
        cs = cs[0].cpu().double()
        for z in (0.6, -1.3):
            # The determinant of the 3 x 3 matrix of the rows' polynomials, each read highest degree first.
            entries = [[_polyval(r[0:4], z), _polyval(r[4:8], z), _polyval(r[8:13], z)] for r in _DET_ROWS]
            det = torch.linalg.det(torch.tensor(entries, dtype=torch.float64))
            # cs[i] multiplies z**i, the reverse of the solve_* layout; read highest degree first it is another
            # polynomial (0.39 instead of 0.94 at z = 0.6).
            ascending = sum(cs[i] * z**i for i in range(11))
            descending = sum(cs[10 - i] * z**i for i in range(11))
            self.assert_close(ascending, det, rtol=1e-6, atol=1e-6)
            assert (descending - det).abs() > 0.1

    @pytest.mark.parametrize(
        "fn, coeffs, expected",
        [
            (solver.solve_cubic, [0.0, 0.0, 2.0, -6.0], [3.0, 0.0, 0.0]),  # 2x - 6
            (solver.solve_cubic, [0.0, 0.0, 1.0, 5.0], [-5.0, 0.0, 0.0]),  # x + 5
            (solver.solve_cubic, [0.0, 1.0, 0.0, -4.0], [2.0, -2.0, 0.0]),  # x^2 - 4
            (solver.solve_cubic, [0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0]),  # 3: no root
            (solver.solve_quartic, [0.0, 0.0, 0.0, 2.0, -6.0], [3.0, 0.0, 0.0, 0.0]),  # through solve_cubic
            (solver.solve_quartic, [0.0, 0.0, 2.0, 0.0, -8.0], [2.0, -2.0, 0.0, 0.0]),  # 2x^2 - 8
            (solver.solve_quadratic, [0.0, 2.0, -6.0], [3.0, 0.0]),  # 2x - 6
            (solver.solve_quadratic, [0.0, -4.0, 2.0], [0.5, 0.0]),  # -4x + 2
            (solver.solve_quadratic, [0.0, 0.0, 5.0], [0.0, 0.0]),  # 5: no root
        ],
        ids=[
            "cubic_linear",
            "cubic_linear_negative_root",
            "cubic_bx2_plus_d",
            "cubic_constant",
            "quartic_linear",
            "quartic_bx2_plus_d",
            "quadratic_linear",
            "quadratic_linear_negative_b",
            "quadratic_constant",
        ],
    )
    def test_convention_zero_leading_coefficient_4873(self, fn, coeffs, expected, device, dtype):
        # #4873: a zero leading coefficient lowers the degree. The roots of the remaining polynomial are
        # returned with the usual 0.0 padding, as numpy.roots does after dropping leading zeros.
        out = fn(torch.tensor([coeffs], device=device, dtype=dtype))
        self.assert_close(out, torch.tensor([expected], device=device, dtype=dtype))

    @pytest.mark.parametrize(
        "fn, coeffs",
        [(solver.solve_quadratic, [0.0, 2.0, -6.0]), (solver.solve_cubic, [0.0, 0.0, 2.0, -6.0])],
        ids=["quadratic_linear", "cubic_linear"],
    )
    def test_convention_zero_leading_coefficient_gradient_4873(self, fn, coeffs, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("Gradient values are checked in float32 and float64.")
        # The root 3 of 2x - 6 keeps its dependence on the zero higher-order coefficients. By the implicit function
        # theorem d root / d coeffs[k] = -root^(n - k) / p'(root), with p'(root) = 2. gradcheck cannot stand in for
        # this: a negative leading coefficient adds real roots, and slot 0 can jump to one of them.
        x = torch.tensor([coeffs], device=device, dtype=dtype, requires_grad=True)
        (grad,) = torch.autograd.grad(fn(x)[0, 0], x)
        powers = [3.0 ** (len(coeffs) - 1 - k) for k in range(len(coeffs))]
        self.assert_close(grad, -torch.tensor([powers], device=device, dtype=dtype) / 2.0)

    @pytest.mark.parametrize(
        "fn, lead", [(solver.solve_quadratic, []), (solver.solve_cubic, [0.0])], ids=["quadratic", "cubic"]
    )
    def test_convention_zero_leading_coefficient_branch_keeps_ordinary_gradients_finite_4873(
        self, fn, lead, device, dtype
    ):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("Gradient values are checked in float32 and float64.")
        # x^2 + b x - 1 with a tiny b is an ordinary quadratic with the roots +-1, but (c / b)^2 overflows.
        # torch.where still differentiates the linear lane it discards for this row, so that lane must not see c.
        b = 1e-20 if dtype == torch.float32 else 1e-160
        x = torch.tensor([[*lead, 1.0, b, -1.0]], device=device, dtype=dtype, requires_grad=True)
        (grad,) = torch.autograd.grad(fn(x).sum(), x)
        # The roots sum to -b / a.
        expected = torch.tensor([[*lead, b, -1.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(grad, expected)

    def test_wart_solve_quartic_small_scale_4833(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("this row's 2e-11 and -4.2e-08 coefficients underflow float16 and keep 3 digits in bfloat16")
        # #4833: x^4 - 0.009x^3 + 3e-05x^2 - 4.2e-08x + 2e-11 = (x - 0.001)(x - 0.002)(x^2 - 0.006x + 1e-05) has
        # the real roots 1e-3 and 2e-3, but the solver is not scale-invariant and returns neither to 1 %.
        coeffs = torch.tensor([[1.0, -0.009, 3e-05, -4.2e-08, 2e-11]], device=device, dtype=dtype)
        out = solver.solve_quartic(coeffs)
        for root in (1e-3, 2e-3):
            assert (out - root).abs().min() > 1e-2 * root

    def test_wart_solve_quartic_absolute_leading_tolerance_4905(self, device, dtype):
        # (x - 1)(x - 2)(x - 3)(x - 4), and the same row times a power of two (exact in every dtype) that brings the
        # leading coefficient below the fallback tolerance (1e-6, or 1e-12 in float64).
        row = torch.tensor([[1.0, -10.0, 35.0, -50.0, 24.0]], device=device, dtype=dtype)
        scale = 2.0**-41 if dtype == torch.float64 else 2.0**-21
        roots = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device, dtype=dtype)
        self.assert_close(solver.solve_quartic(row).sort(dim=-1).values, roots)
        # #4905: the tolerance is absolute, so the scaled row is solved as the cubic that remains without x^4, and
        # the roots 2, 3 and 4 are lost. Once the test is relative, both rows give the same roots.
        out = solver.solve_quartic(row * scale)
        for root in (2.0, 3.0, 4.0):
            assert (out - root).abs().min() > 0.5
