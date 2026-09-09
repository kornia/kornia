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
            (torch.tensor([[0.0, 0.0, 1.0, -1.0]]), torch.tensor([[1.0, 0.0, 0.0]])),  # handle first order
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

    @pytest.mark.parametrize(
        ("coeffs", "expected_real"),
        [
            # A resolvent cubic with one negative real root: the y=0 placeholder used to win the
            # R^2 argmax and return a spurious -2 while missing both real roots (#4346).
            ([1.0, 2.0, 0.0, 0.0, -16.0], [-2.760555138195215, 1.6383450267923287]),
            # A biquadratic: R^2 is 0, so R = sqrt(±1e-16) ≈ 1e-8 previously beat the absolute
            # tolerance on R and sent the division path a near-zero denominator (#4346).
            ([1.0, 0.0, -4.0, 0.0, -5.0], [-2.23606797749979, 2.23606797749979]),
            # A 1e-6 x^3 perturbation of x^4 - 16 must leave the ±2 roots in place, not vanish them.
            ([1.0, 1e-6, 0.0, 0.0, -16.0], [-2.0, 2.0]),
        ],
    )
    def test_solve_quartic_recovers_real_roots_4346(self, coeffs, expected_real, device):
        roots = solver.solve_quartic(torch.tensor([coeffs], device=device, dtype=torch.float64))
        real = torch.sort(roots[0][roots[0].abs() > 1e-12]).values
        expected = torch.sort(torch.tensor(expected_real, device=device, dtype=torch.float64)).values
        self.assert_close(real, expected, rtol=1e-6, atol=1e-6)

    def test_random_general_coefficients_match_numpy_roots_4346(self, device):
        # ``test_random`` only builds quartics from four *real* roots, so its resolvent cubic always
        # has three real roots and neither defective branch is reached. Draw general coefficients and
        # check that every returned entry is a real root and that every real root is returned.
        torch.manual_seed(0)
        batch_size = 64
        coeffs = torch.rand(batch_size, 5, device=device, dtype=torch.float64) * 10 - 5
        coeffs[:, 0] = 1.0
        roots = solver.solve_quartic(coeffs).detach().cpu().numpy()
        coeffs_np = coeffs.cpu().numpy()
        for i in range(batch_size):
            reference = np.roots(coeffs_np[i])
            real_ref = reference[np.abs(reference.imag) < 1e-7].real
            # no spurious entries: every returned root is a genuine real root
            for g in roots[i]:
                if abs(g) < 1e-12:
                    continue
                assert np.any(np.abs(g - real_ref) < 1e-6), (coeffs_np[i], g, real_ref)
            # nothing missing: every real root is returned
            for t in real_ref:
                assert np.any(np.abs(roots[i] - t) < 1e-6), (coeffs_np[i], t, roots[i])

    @pytest.mark.parametrize(
        ("coeffs", "expected", "expected_grad"),
        [
            # x^4 - 16 = (x^2-4)(x^2+4): two real roots, +-2. R_sq lands exactly on 0.
            ([1.0, 0.0, 0.0, 0.0, -16.0], [2.0, -2.0], [0.0, -0.5, 0.0, 0.0, 0.0]),
            # x^4 - 1 = (x^2-1)(x^2+1): two real roots, +-1. R_sq lands exactly on 0.
            ([1.0, 0.0, 0.0, 0.0, -1.0], [1.0, -1.0], [0.0, -0.5, 0.0, 0.0, 0.0]),
            # (x^2+1)(x^2+4): no real roots. R_sq < 0 sends R to the `R approx 0` fallback,
            # whose own radicand is then exactly 0 -- the second sqrt site.
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
        # convention, not a Jacobian -- the forward is discontinuous at exactly these points
        # (#4346), so there is no finite-difference derivative to pin against.
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
