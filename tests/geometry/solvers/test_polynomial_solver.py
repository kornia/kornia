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
from kornia.geometry.solvers.polynomial_solver import _exact_power_of_two, _solve_cubic_real, _solve_cubic_with_count

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

    def test_stable_formula_4914(self, device, dtype):
        # #4914: the direct quadratic formula loses the finite root through cancellation
        # when |4ac| << b^2. The stable formulation should retain both real roots.
        if dtype == torch.float16:
            pytest.skip("1e-8 rounds to 0 in float16, and the root near -2e8 is beyond its range")
        coeffs = torch.tensor([[1e-8, 2.0, -6.0]], device=device, dtype=dtype)
        roots = solver.solve_quadratic(coeffs)

        expected = torch.tensor([3.0, -2e8], device=device, dtype=dtype)
        self.assert_close(roots[0], expected, rtol=1e-6, atol=1e-5)

    def test_stable_formula_gradient_4914(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("Gradient values are checked in float32 and float64.")
        # By the implicit function theorem d root / d coeffs[k] = -root^(2 - k) / p'(root), with p'(r) = 2ar + b.
        # (-b + sqrt(D)) / (2a) loses the root near 3 of 1e-8 x^2 + 2x - 6 to cancellation, and its gradient with
        # it: d root / da came out as 3e8 in float32 and -4.96 instead of -4.5 in float64.
        x = torch.tensor([[1e-8, 2.0, -6.0]], device=device, dtype=dtype, requires_grad=True)
        roots = solver.solve_quadratic(x)
        for slot in range(2):
            (grad,) = torch.autograd.grad(roots[0, slot], x, retain_graph=True)
            r = roots[0, slot].detach()
            expected = -torch.stack([r * r, r, torch.ones_like(r)]) / (2e-8 * r + 2.0)
            self.assert_close(grad[0], expected, rtol=1e-5, atol=0.0)

    def test_stable_formula_keeps_order_and_edge_gradients_4914(self, device, dtype):
        # Slot 0 is (-b + sqrt(D)) / (2a) also for b == 0, where the sign of b does not pick the branch.
        out = solver.solve_quadratic(torch.tensor([[1.0, 0.0, -4.0], [-1.0, 0.0, 4.0]], device=device, dtype=dtype))
        self.assert_close(out, torch.tensor([[2.0, -2.0], [-2.0, 2.0]], device=device, dtype=dtype))
        if dtype not in (torch.float32, torch.float64):
            return
        # c / q is only taken where the two real roots differ. With no real root and a tiny b, c / q^2 overflows,
        # and torch.where would turn the row's zero gradient into nan. At the double root of x^2 - 6x + 9 both
        # slots are -b / (2a), so the gradient of their sum is that of -b / a, [b / a^2, -1 / a, 0].
        tiny = 1e-25 if dtype == torch.float32 else 1e-160
        x = torch.tensor([[1.0, tiny, 1.0], [1.0, -6.0, 9.0]], device=device, dtype=dtype, requires_grad=True)
        (grad,) = torch.autograd.grad(solver.solve_quadratic(x).sum(), x)
        self.assert_close(grad, torch.tensor([[0.0, 0.0, 0.0], [-6.0, -1.0, 0.0]], device=device, dtype=dtype))

    @pytest.mark.parametrize(
        "coeffs, expected, literal_dtype",
        [
            # b^2 underflows, but it is negligible against 4ac: rescaling by the largest coefficient flushed a to 0.
            ([2.0**-45, 2.0**-121, -(2.0**105)], [2.0**75, -(2.0**75)], "float32"),
            ([2.0**-500, 2.0**-1000, -(2.0**600)], [2.0**550, -(2.0**550)], "float64"),
            # No real root; the rescaled row lost its subnormal a and became a linear equation with a root at -2.6e33.
            ([-2.129973665773722e-43, -6.776749542384429e-25, -1768312192.0], [0.0, 0.0], "float32"),
        ],
    )
    def test_underflowed_negligible_term_keeps_small_coefficients(self, coeffs, expected, literal_dtype, device, dtype):
        if dtype != getattr(torch, literal_dtype) or device.type != "cpu":
            pytest.skip("The literal needs exact CPU subnormal and extreme-exponent arithmetic in its own dtype.")
        roots = solver.solve_quadratic(torch.tensor([coeffs], device=device, dtype=dtype))
        self.assert_close(roots, torch.tensor([expected], device=device, dtype=dtype), rtol=1e-6, atol=0.0)


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

    def test_float32_close_pair_does_not_depend_on_the_batch(self, device, dtype):
        # The float32 Cardano discriminant of this row takes the wrong sign at the close pair near 3.62, and an
        # uncertainty bound on |Q^3| + R^2 misses the cancellation inside Q and R. Solved alone it returned one root;
        # next to a row that forced float64, all three.
        if dtype != torch.float32 or device.type != "cpu":
            pytest.skip("CPU solves every float32 cubic in float64; eager accelerators promote flagged rows only.")
        row = [1.0, -9.652440071105957, 30.569580078125, -31.61037254333496]
        alone = solver.solve_cubic(torch.tensor([row], device=device, dtype=dtype))
        # Roots of the represented float32 coefficients, solved in float64.
        expected = torch.tensor([[2.411706999775569, 3.619729296817037, 3.621003774513351]], device=device, dtype=dtype)
        self.assert_close(alone.sort(-1).values, expected, rtol=1e-6, atol=0.0)
        flagged = [1.0, -481.0438232421875, 57850.8359375, -11.54153823852539]
        batched = solver.solve_cubic(torch.tensor([row, flagged], device=device, dtype=dtype))
        self.assert_close(batched[:1], alone, rtol=0.0, atol=0.0)

    def test_root_beside_a_complex_pair_is_not_taken_as_dominant(self, device, dtype):
        # x (x^2 + x + 3) has one real root, 0. The closed form returns a cancellation remnant of about eps there, and
        # treating it as a dominant root made Vieta's quotients by it report a spurious real pair of size 1 / eps.
        rows = [[1.0, 1.0, 3.0, 0.0], [1.0, 1.0, 1.0, 0.0]]
        if dtype in (torch.float32, torch.float64):
            rows.append([1.0, 1.0, 1.0, -1e-30])  # (x - 1e-30) (x^2 + x + 1), up to rounding
        roots, count = _solve_cubic_with_count(torch.tensor(rows, device=device, dtype=dtype))
        assert count.tolist() == [1] * len(rows)
        atol = 8 * torch.finfo(dtype).eps
        self.assert_close(roots, torch.zeros_like(roots), rtol=0.0, atol=atol)

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

    def test_real_root_count_4862(self, device, dtype):
        # #4862: the 0.0 padding is indistinguishable from a root at 0 in the roots alone; the count tells them apart.
        coeffs = torch.tensor(
            [
                [1.0, -6.0, 11.0, -6.0],  # (x - 1)(x - 2)(x - 3): D <= 0
                [1.0, 0.0, 1.0, 0.0],  # x (x^2 + 1): one real root at 0, D > 0
                [1.0, 0.0, 0.0, -1.0],  # x^3 - 1: Q == 0
                [1.0, 3.0, 3.0, 1.0],  # (x + 1)^3: Q == R == 0
                [1.0, -2.0, 1.0, 0.0],  # x (x - 1)^2: D == 0, the double root counts twice
                [0.0, 1.0, -3.0, 2.0],  # (x - 1)(x - 2)
                [0.0, 1.0, -2.0, 1.0],  # (x - 1)^2: zero discriminant, the double root counts twice
                [0.0, 1.0, 0.0, 1.0],  # x^2 + 1
                [0.0, 0.0, 2.0, -1.0],  # 2x - 1
                [0.0, 0.0, 0.0, 1.0],  # a nonzero constant
            ],
            device=device,
            dtype=dtype,
        )
        roots, num_real = _solve_cubic_with_count(coeffs)
        self.assert_close(roots, solver.solve_cubic(coeffs), rtol=0.0, atol=0.0)
        assert num_real.tolist() == [3, 1, 1, 3, 3, 2, 2, 0, 1, 0]

    def test_tiny_leading_coefficient_4914(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip("1e-30 rounds to 0 in float16, which solves the row as the linear equation 2x - 6 = 0")
        # #4914: 1e-30 x^3 + 2x - 6 has the real root 3 (the other two are +-1.4e15 i). Normalising by a gave
        # c/a = 2e30 and d/a = -6e30, whose Q^3 overflowed float32 (root inf), and in float64 Cardano's A + B
        # cancelled to 0 (root 0). The row is now solved scaled by 2^-51, and A + B as 2R / (A^2 + B^2 + Q).
        coeffs = torch.tensor([[1e-30, 0.0, 2.0, -6.0]], device=device, dtype=dtype)
        roots = solver.solve_cubic(coeffs)
        expected = torch.tensor([[3.0, 0.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(roots, expected, rtol=8 * torch.finfo(dtype).eps, atol=0.0)

    def test_tiny_leading_coefficient_gradient_4914(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("Gradient values are checked in float32 and float64.")
        # Implicit-function derivative of the simple root r: dr/dcoeffs[k] = -r^(3 - k) / p'(r), p'(3) = 2 here.
        # The root moves by 27a/2 with a and by 9b/2 with b, below the resolution of the closed form at a = 1e-30,
        # so those two derivatives cannot come out of it (inf in float32); the c and d derivatives do.
        x = torch.tensor([[1e-30, 0.0, 2.0, -6.0]], device=device, dtype=dtype, requires_grad=True)
        (grad,) = torch.autograd.grad(solver.solve_cubic(x)[0, 0], x)
        self.assert_close(grad[0, 2:], torch.tensor([-1.5, -0.5], device=device, dtype=dtype), rtol=1e-5, atol=0.0)

        # The root bound takes |d|^(1/3), whose derivative is infinite at d == 0. The bound is a step function of
        # the coefficients and is detached: the root 0 of x^3 + x^2 + x keeps its derivative -1 / p'(0) in d.
        x = torch.tensor([[1.0, 1.0, 1.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        (grad,) = torch.autograd.grad(solver.solve_cubic(x)[0, 0], x)
        self.assert_close(grad, torch.tensor([[0.0, 0.0, 0.0, -1.0]], device=device, dtype=dtype), rtol=0.0, atol=1e-6)

        # 2x^3 has the root bound 0 and is not scaled. Its triple root 0 keeps the Q == R == 0 branch's gradient,
        # -b / (3a) per root, so the sum of the roots moves by -1 / a with b.
        x = torch.tensor([[2.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        roots = solver.solve_cubic(x)
        self.assert_close(roots.detach(), torch.zeros(1, 3, device=device, dtype=dtype), rtol=0.0, atol=0.0)
        (grad,) = torch.autograd.grad(roots.sum(), x)
        self.assert_close(grad, torch.tensor([[0.0, -0.5, 0.0, 0.0]], device=device, dtype=dtype), rtol=0.0, atol=0.0)

    def test_scaled_row_has_scaled_roots_4914(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("Half-precision rows are solved in float32, and 2^40 is beyond float16.")
        # The row scale has to be an exact power of two on every backend: torch.exp2 is not, for integer arguments
        # on MPS, and the triple root below then left its Q == R == 0 branch.
        exponents = torch.tensor([-120.0, -38.0, 0.0, 2.0, 120.0], device=device, dtype=dtype)
        expected = torch.tensor([2.0**-120, 2.0**-38, 1.0, 4.0, 2.0**120], device=device, dtype=dtype)
        assert torch.equal(_exact_power_of_two(exponents), expected), _exact_power_of_two(exponents)

        # Coefficient i divided by s^i moves every root by 1 / s. With s an exact power of two the scaled row reaches
        # the closed form as the same numbers, so its roots are the original ones times 1 / s. One row per branch:
        # three real roots, one real root with Q > 0 and with Q < 0, Q == 0, and a triple root. Before the fix,
        # Q^3 left the dtype's range at these s and the rows returned zeros, inf or a wrong branch.
        rows = torch.tensor(
            [
                [1.0, -6.0, 11.0, -6.0],  # (x - 1)(x - 2)(x - 3)
                [1.0, 0.0, 1.0, -2.0],  # (x - 1)(x^2 + x + 2): Q > 0
                [1.0, 3.0, 1.0, -5.0],  # (x - 1)(x^2 + 4x + 5): Q < 0
                [1.0, 0.0, 0.0, 1.0],  # x^3 + 1: Q == 0
                [1.0, -3.0, 3.0, -1.0],  # (x - 1)^3
            ],
            device=device,
            dtype=dtype,
        )
        base = solver.solve_cubic(rows)
        exponent = 40 if dtype == torch.float32 else 200
        for s in (2.0**exponent, 2.0**-exponent):
            # Python computes the powers of two exactly, and multiplying by them is exact on every backend, where
            # `s ** torch.arange(4)` and a division are not (MPS).
            powers = torch.tensor([1.0, 1.0 / s, 1.0 / s**2, 1.0 / s**3], device=device, dtype=dtype)
            self.assert_close(solver.solve_cubic(rows * powers) * s, base, rtol=8 * torch.finfo(dtype).eps, atol=0.0)

    def test_small_r_keeps_the_root_4914(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("2e-17 rounds to 0 in float16, and bfloat16 keeps 3 digits of these rows.")
        # The D > 0 branch took A = B = 0 for |R| <= 1e-16 and returned -b / 3 for the root. x^3 + x - 2e-17 has
        # R = 1e-17 and the root 2e-17, not 0. The second row is the resolvent cubic of the #4833 quartic
        # x^4 - 0.009x^3 + 3e-05x^2 - 4.2e-08x + 2e-11: its R is 2e-18 and its real root 1.2e-5, which came back
        # as -b / 3 = 1e-5 and lost the quartic's roots 1e-3 and 2e-3.
        rows = torch.tensor([[1.0, 0.0, 1.0, -2e-17], [1.0, -3e-05, 2.98e-10, -9.84e-16]], device=device, dtype=dtype)
        roots = solver.solve_cubic(rows)
        expected = torch.tensor([[2e-17, 0.0, 0.0], [1.2e-5, 0.0, 0.0]], device=device, dtype=dtype)
        tol = 1e-5 if dtype == torch.float32 else 1e-12
        self.assert_close(roots, expected, rtol=tol, atol=0.0)

    def test_root_bound_takes_every_coefficient_4914(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("These rows are beyond the range of float16, and half-precision rows are solved in float32.")
        # Each row's root bound is set by one coefficient: b (x^3 + b x^2 = x^2 (x + b)), c (the root -d / c of
        # x^3 + c x + d with c huge) and b at the top of the dtype's range, where the scale exponent has to be
        # clamped. Scaled without that coefficient's term in the bound, or with an unclamped exponent, Q^3 and R^2
        # overflow or the scale is not a finite power of two, and the root comes back as 0 or nan.
        if dtype == torch.float32:
            rows = [[1.0, 1e20, 0.0, 0.0], [1.0, 0.0, 1e30, 1e20], [1.0, 1e38, 0.0, 0.0]]
            expected = [[-1e20, 0.0, 0.0], [-1e-10, 0.0, 0.0], [-1e38, 0.0, 0.0]]
        else:
            rows = [[1.0, 1e160, 0.0, 0.0], [1.0, 0.0, 1e200, 1.0], [1.0, 1e308, 0.0, 0.0]]
            expected = [[-1e160, 0.0, 0.0], [-1e-200, 0.0, 0.0], [-1e308, 0.0, 0.0]]
        roots = solver.solve_cubic(torch.tensor(rows, device=device, dtype=dtype))
        expected = torch.tensor(expected, device=device, dtype=dtype)
        self.assert_close(roots.sort(dim=-1).values, expected, rtol=8 * torch.finfo(dtype).eps, atol=0.0)

    def test_dominant_root_keeps_the_other_two_4914(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("1e-30 rounds to 0 in float16, and half-precision rows are solved in float32.")
        # #4914: with b != 0, a tiny a puts a root near -b / a in front of the quadratic b x^2 + c x + d. The scaled
        # row's D = Q^3 + R^2 is then a difference of nearly equal numbers, and the closed form either dropped the two
        # small real roots or returned values that are not roots in their slots (6.65e3 and -6.64e3 for a = 1e-8 in
        # float32). The dominant root stays and the other two come from Vieta's relations and the stable quadratic,
        # whose discriminant tells the real pair 1 + a + O(a^2), 2 - 8a + O(a^2) from the complex pair of x^2 + x + 1.
        tol = 1e-6 if dtype == torch.float32 else 1e-13
        for a in (1e-8, 1e-30):
            rows = torch.tensor([[a, 1.0, -3.0, 2.0], [a, 1.0, 1.0, 1.0]], device=device, dtype=dtype)
            expected = torch.tensor(
                [[-1 / a - 3, 1 + a, 2 - 8 * a], [-1 / a + 1, 0.0, 0.0]], device=device, dtype=dtype
            )
            roots, num_real = _solve_cubic_with_count(rows)
            self.assert_close(roots[:1].sort(dim=-1).values, expected[:1].sort(dim=-1).values, rtol=tol, atol=0.0)
            # Unsorted: the single real root goes to slot 0 and the count marks slots 1 and 2 as padding. The closed
            # form had put -1 / a in slot 1 here (float32 at a = 1e-8, float64 at a = 1e-30).
            self.assert_close(roots[1:], expected[1:], rtol=tol, atol=0.0)
            assert num_real.tolist() == [3, 1], num_real

    def test_dominant_root_over_a_double_zero_root_4914(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("1e-30 rounds to 0 in float16, and half-precision rows are solved in float32.")
        # x^2 (x + b / a): Vieta gives the pair total = product = 0, a zero discriminant, so the double root 0 is real
        # and counts twice, as solve_quadratic's delta == 0 double root and the D == 0 row of #4862 do.
        rows = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1e-30, 1.0, 0.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([[-1.0, 0.0, 0.0], [-1e30, 0.0, 0.0]], device=device, dtype=dtype)
        roots, num_real = _solve_cubic_with_count(rows)
        self.assert_close(roots.sort(dim=-1).values, expected, rtol=8 * torch.finfo(dtype).eps, atol=0.0)
        assert num_real.tolist() == [3, 3], num_real

    def test_dominant_root_gradcheck_4914(self, device):
        # Vieta rows: (x - 1)(x - 2)(x + 1000), (x + 1000)(x^2 + x + 1) and (x - 32)(x - 1)(x + 1). The first and the
        # last were right before and keep the closed form's slot order; the complex pair puts its root in slot 0.
        rows = torch.tensor(
            [[1.0, 997.0, -2998.0, 2000.0], [1.0, 1001.0, 1001.0, 1000.0], [1.0, -32.0, -1.0, 32.0]],
            device=device,
            dtype=torch.float64,
        )
        expected = torch.tensor(
            [[2.0, -1000.0, 1.0], [-1000.0, 0.0, 0.0], [32.0, -1.0, 1.0]], device=device, dtype=torch.float64
        )
        roots, num_real = _solve_cubic_with_count(rows)
        self.assert_close(roots, expected, rtol=1e-13, atol=0.0)
        assert num_real.tolist() == [3, 1, 3], num_real
        self.gradcheck(solver.solve_cubic, (rows.requires_grad_(),))

    def test_tiny_leading_coefficient_gradient_stays_finite(self, device, dtype):
        if dtype != torch.float64:
            pytest.skip("The coefficients need the float64 exponent range.")
        coeffs = torch.tensor(
            [[-3.261706204944339e-40, -2.3121190290020114e101, 9.353450215509385e-208, -3.1729113405545114e257]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        root = solver.solve_cubic(coeffs)[0, 0]
        (gradient,) = torch.autograd.grad(root, coeffs)
        # The implicit derivative -[r^3, r^2, r, 1] / p'(r), from mpmath at 50 digits. The quotient b / a has the
        # derivative b / a^2, which overflows; the root's derivative with respect to a does not.
        expected = torch.tensor(
            [[-2.173304145468546e180, 3.0658800553039539e39, -4.325036849126378e-102, 6.1013292786649817e-243]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(gradient, expected, atol=0.0, rtol=1e-8)

    def test_exact_double_root(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("The coefficients are exact in float32 and float64.")
        # (x + 5.25)(x - 3.125)^2: a rounding-level positive discriminant reported the single root -5.25.
        coeffs = torch.tensor([[1.0, -1.0, -23.046875, 51.26953125]], device=device, dtype=dtype)
        roots, num_real = _solve_cubic_with_count(coeffs)
        expected = torch.tensor([[-5.25, 3.125, 3.125]], device=device, dtype=dtype)
        self.assert_close(roots.sort(-1).values, expected, atol=0.0, rtol=0.0)
        assert num_real.tolist() == [3]

    def test_exact_double_root_grid(self, device, dtype):
        # (x - r)^2 (x - s) for every pair of distinct nonzero quarter-integers in [-6, 6]; coefficients exact.
        values = torch.arange(-24, 25, dtype=torch.float64) / 4
        values = values[values != 0]
        r, s = (v.flatten() for v in torch.meshgrid(values, values, indexing="ij"))
        r, s = r[r != s], s[r != s]
        coeffs = torch.stack([torch.ones_like(r), -(2 * r + s), r * r + 2 * r * s, -r * r * s], -1)
        exact = (coeffs.to(dtype).double() == coeffs).all(-1)
        coeffs = coeffs[exact].to(device=device, dtype=dtype)
        expected = torch.stack([r, r, s], -1)[exact].sort(-1).values.to(device=device, dtype=dtype)
        roots, num_real = _solve_cubic_with_count(coeffs)
        assert bool((num_real == 3).all())
        self.assert_close(roots.sort(-1).values, expected, atol=1e-6, rtol=1e-6)

    def test_double_root_beside_a_rounding_level_discriminant(self, device, dtype):
        if dtype != torch.float64:
            pytest.skip("The coefficients are exact in float64.")
        # A close real pair whose discriminant is 4e-19 of its terms, beyond 2 eps but within the 32 eps window
        # in which the stationary points decide. Reference: sympy real-root isolation.
        coeffs = torch.tensor([[1.0, -6.566669464111328, 12.517459229177803, -7.397783069339488]], device=device, dtype=dtype)
        roots, num_real = _solve_cubic_with_count(coeffs)
        expected = torch.tensor([[1.4022817537442716, 1.4022817685945957, 3.7621059417724609]], device=device, dtype=dtype)
        assert num_real.tolist() == [3]
        self.assert_close(roots.sort(-1).values, expected, atol=0.0, rtol=1e-7)

    def test_exact_double_root_gradient(self, device, dtype):
        if dtype != torch.float64:
            pytest.skip("The Jacobian is compared in float64.")
        coeffs = torch.tensor([[1.0, -1.0, -23.046875, 51.26953125], [2.0, -10.0, 16.0, -8.0]], device=device, dtype=dtype)
        coeffs.requires_grad_()
        roots = solver.solve_cubic(coeffs)
        (gradient,) = torch.autograd.grad(roots.sum(), coeffs)
        # A repeated root's Jacobian is undefined; the surrogate stays finite and keeps the sum of the roots, -b / a.
        expected = torch.zeros_like(gradient)
        expected[:, 0] = coeffs.detach()[:, 1] / coeffs.detach()[:, 0].square()
        expected[:, 1] = -1 / coeffs.detach()[:, 0]
        self.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)


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
    @pytest.mark.parametrize(
        "expected",
        [
            [-3.0, -2.5, -2.5, -1.5],  # Newton moved a copy of -2.5 onto -1.5 (#5509/#5621).
            [-6.0, -5.0, -5.0, -3.0],
            [-6.0, -6.0, -5.75, -3.5],  # A negative factor discriminant discarded the double root.
            [-6.0, -5.75, 4.25, 4.25],
            [-6.0, -5.0, -5.0, -3.75],  # Small R amplifies the factor coefficient roundoff.
            [-1.75, 1.0, 1.0, 6.0],
        ],
    )
    @pytest.mark.parametrize("scale", [2.0**-20, 1.0, -(2.0**20)])
    def test_exact_double_root_multiplicity_5622(self, expected, scale, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("These exact coefficient and root comparisons require float32 or float64.")
        roots = torch.tensor([expected], dtype=torch.float64)
        coeffs = (_monic_from_roots(roots) * scale).to(device=device, dtype=dtype)
        actual = solver.solve_quartic(coeffs)
        assert actual.dtype == dtype
        assert bool((actual != 0).all())
        # Sorting all four slots tests multiplicity as well as presence: neither padding nor
        # moving a copy onto a simple root can satisfy this bound (the smallest gap is 0.25).
        self.assert_close(actual.sort(-1).values, roots.to(device=device, dtype=dtype), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("delta", [2.0**-52, 2.0**-50])
    def test_near_double_complex_roots_are_not_recovered_5622(self, delta, device, dtype):
        if dtype != torch.float64:
            pytest.skip("The perturbation must remain representable in the input coefficients.")
        # (x^2 - 1)^2 + delta*(x^2 + 1) is strictly positive. Both stationary
        # points satisfy the old residual/derivative bounds but are not real roots.
        coeffs = torch.tensor([[1.0, 0.0, -2.0 + delta, 0.0, 1.0 + delta]], device=device, dtype=dtype)
        actual = solver.solve_quartic(coeffs)
        self.assert_close(actual, torch.zeros_like(actual), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("scale", [2.0**-900, 1.0, -(2.0**900)])
    def test_double_root_recovery_real_complex_boundary_5622(self, scale, device, dtype):
        if dtype != torch.float64:
            pytest.skip("The perturbation and coefficient scales require float64.")
        # ((x - 1)^2 + delta)*(x + 1)*(x + 2) has only two real roots.
        # The second row needs recovery of an exact double root, even at extreme scales.
        delta = 2.0**-50
        coeffs = (
            torch.tensor(
                [
                    [1.0, 1.0, -3.0 + delta, -1.0 + 3.0 * delta, 2.0 + 2.0 * delta],
                    [1.0, 3.25, -47.3125, -81.015625, 623.15625],
                ],
                device=device,
                dtype=dtype,
            )
            * scale
        )
        expected = torch.tensor([[-2.0, -1.0, 0.0, 0.0], [-6.0, -5.75, 4.25, 4.25]], device=device, dtype=dtype)
        self.assert_close(solver.solve_quartic(coeffs).sort(-1).values, expected, atol=1e-5, rtol=1e-5)

    def test_double_root_controls_5622(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("Close distinct-root controls require float32 or float64.")
        expected = torch.tensor(
            [[-3.0, -2.5, -2.375, -1.5], [-6.0, -5.75, 4.125, 4.25], [-6.0, -1.75, 1.0, 1.125]],
            dtype=torch.float64,
        )
        coeffs = _monic_from_roots(expected).to(device=device, dtype=dtype)
        self.assert_close(
            solver.solve_quartic(coeffs).sort(-1).values, expected.to(device=device, dtype=dtype), atol=1e-5, rtol=1e-5
        )
        # (x + 6)(x + 5.75)((x - 4.25)^2 + 1/16): its nearby complex pair must stay padding.
        coeffs = torch.tensor([[1.0, 3.25, -47.25, -80.28125, 625.3125]], device=device, dtype=dtype)
        expected = torch.tensor([[-6.0, -5.75, 0.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(solver.solve_quartic(coeffs).sort(-1).values, expected, atol=1e-5, rtol=1e-5)

    def test_double_root_gradient_and_dtype_5622(self, device, dtype):
        # The first row is exact in float16; the second is also exact in bfloat16.
        rows = [[1.0, 9.5, 33.25, 50.625, 28.125], [1.0, 0.0, -9.0, 4.0, 12.0]]
        expected = [[-3.0, -2.5, -2.5, -1.5], [-3.0, -1.0, 2.0, 2.0]]
        if dtype == torch.bfloat16:
            rows, expected = rows[1:], expected[1:]
        elif dtype in (torch.float32, torch.float64):
            rows.append([1.0, 3.25, -47.3125, -81.015625, 623.15625])
            expected.append([-6.0, -5.75, 4.25, 4.25])
        coeffs = torch.tensor(rows, device=device, dtype=dtype, requires_grad=True)
        actual = solver.solve_quartic(coeffs)
        assert actual.dtype == dtype
        self.assert_close(
            actual.sort(-1).values, torch.tensor(expected, device=device, dtype=dtype), atol=1e-5, rtol=1e-5
        )
        actual.sum().backward()
        assert bool(torch.isfinite(coeffs.grad).all())
        # The sum of the four real roots is -b/a. A detached-output fix fails this derivative.
        expected_grad = torch.zeros_like(coeffs)
        expected_grad[:, 0] = coeffs.detach()[:, 1]
        expected_grad[:, 1] = -1.0
        self.assert_close(coeffs.grad, expected_grad, atol=1e-4, rtol=1e-4)

    def test_double_root_dynamo_5622(self, device, dtype, torch_optimizer):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("The issue coefficients are exact in float32 and float64.")
        coeffs = torch.tensor(
            [[1.0, 9.5, 33.25, 50.625, 28.125], [1.0, 3.25, -47.3125, -81.015625, 623.15625]],
            device=device,
            dtype=dtype,
        )
        compiled = torch_optimizer(solver.solve_quartic)
        self.assert_close(compiled(coeffs), solver.solve_quartic(coeffs), atol=1e-5, rtol=1e-5)

    def test_exact_double_root_grid_5622(self, device, dtype):
        # All 51,888 quartics with one double root and two different simple roots
        # on the nonzero quarter-integer grid [-6, 6]. Coefficients are binary-exact
        # in float32/64; half coverage retains only inputs exactly representable there.
        values = torch.arange(-24, 25, dtype=torch.float64) / 4
        values = values[values != 0]
        pairs = torch.combinations(values, r=2)
        doubled = values[:, None].expand(-1, len(pairs)).flatten()
        simple = pairs.repeat(len(values), 1)
        keep = (simple != doubled[:, None]).all(-1)
        roots = torch.cat([doubled[:, None].repeat(1, 2), simple], -1)[keep]
        coefficients = _monic_from_roots(roots)
        exact = (coefficients.to(dtype).double() == coefficients).all(-1)
        coefficients = coefficients[exact].to(device=device, dtype=dtype)
        expected = roots[exact].sort(-1).values.to(device=device, dtype=dtype)
        actual = solver.solve_quartic(coefficients).sort(-1).values
        assert actual.shape == expected.shape
        self.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    @pytest.mark.parametrize(
        "coefficients, expected",
        [
            ([1.0, -6.25, -1.625, 0.0, 0.0], [6.5, 0.0, 0.0, -0.25]),  # x^2 (x - 6.5)(x + 0.25)
            ([1.0, -2.0, -3.0, 0.0, 0.0], [3.0, 0.0, 0.0, -1.0]),
            ([2.0, -4.0, 2.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]),  # 2 x^2 (x - 1)^2
            ([1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),  # x^2 (x^2 + 1): the pair is padding
            ([1.0, -3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]),
            ([-0.5, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
        ],
    )
    def test_exact_zero_roots(self, coefficients, expected, device, dtype):
        # A trailing pair of zero coefficients is an exact multiple root at 0: it is reported exactly, and the
        # zeros sort with the genuine roots.
        coeffs = torch.tensor([coefficients], device=device, dtype=dtype)
        actual = solver.solve_quartic(coeffs)
        self.assert_close(actual, torch.tensor([expected], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_exact_zero_double_root_grid(self, device, dtype):
        # x^2 (x - p)(x - q) for every pair of distinct nonzero quarter-integers in [-6, 6].
        values = torch.arange(-24, 25, dtype=torch.float64) / 4
        pairs = torch.combinations(values[values != 0], r=2)
        roots = torch.cat([pairs, torch.zeros_like(pairs)], -1)
        coefficients = _monic_from_roots(roots)
        exact = (coefficients.to(dtype).double() == coefficients).all(-1)
        coefficients = coefficients[exact].to(device=device, dtype=dtype)
        expected = roots[exact].sort(-1, descending=True).values.to(device=device, dtype=dtype)
        self.assert_close(solver.solve_quartic(coefficients), expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "coefficients, expected, dtypes",
        [
            # Exact roots 512, 2^-13, 2^-18 and -2^19.
            (
                [1.0, 523775.999874115, -268435521.9355469, 33792.00024390221, -0.125],
                [512.0, 2.0**-13, 2.0**-18, -(2.0**19)],
                (torch.float64,),
            ),
            (
                [1.0, 499500.0, -250000048.0, 26000.0, -0.10000000149011612],
                [499.99999200799115, 0.000100000000750412, 4.0000000927561731e-6, -500000.00009600799],
                (torch.float32, torch.float64),
            ),
            # The two small resolvent roots came back with the wrong sign.
            (
                [1.0, 19749794.506627306, -268558166469888.28, 705704923847.0411, 57874906.04389983],
                [9258104.658492141, 0.0027073533624295231, -7.9598886503736466e-5, -29007899.167747202],
                (torch.float64,),
            ),
        ],
    )
    def test_roots_spanning_many_decades(self, coefficients, expected, dtypes, device, dtype):
        if dtype not in dtypes:
            pytest.skip("The coefficients are exact in the listed dtypes only.")
        # References: sympy real-root isolation of the represented coefficients. Four real roots
        # span up to 11 decades; the resolvent's two small roots sit as far below its third.
        coeffs = torch.tensor([coefficients], device=device, dtype=dtype)
        expected = torch.tensor([expected], device=device, dtype=dtype)
        self.assert_close(solver.solve_quartic(coeffs), expected, atol=0.0, rtol=1e-6)

    @pytest.mark.parametrize(
        "coefficients, expected, dtypes",
        [
            # (x - 1)(x - 3)((x - 1)^2 + 2^-24)
            ([1.0, -6.0, 12.000000059604645, -10.000000238418579, 3.0000001788139343], [3.0, 1.0], (torch.float64,)),
            # (x - 2)(x - 3)((x - 2)^2 + 2^-18)
            (
                [1.0, -9.0, 30.000003814697266, -44.00001907348633, 24.000022888183594],
                [3.0, 2.0],
                (torch.float32, torch.float64),
            ),
        ],
    )
    def test_simple_root_beside_a_close_complex_pair(self, coefficients, expected, dtypes, device, dtype):
        if dtype not in dtypes:
            pytest.skip("The coefficients are exact in the listed dtypes only.")
        # The pair makes the simple root's slope small, yet the root's Ferrari partner is the other real root.
        coeffs = torch.tensor([coefficients], device=device, dtype=dtype)
        expected = torch.tensor([expected + [0.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(solver.solve_quartic(coeffs), expected, atol=1e-6, rtol=1e-6)

    def test_simple_root_beside_a_close_complex_pair_grid(self, device, dtype):
        if dtype != torch.float64:
            pytest.skip("The grid coefficients are exact in float64.")
        # (x - r)(x - s)((x - r)^2 + d^2) on dyadic r, s and d: exactly two real roots.
        r = torch.tensor([-2.5, -0.75, 1.0, 3.25], dtype=torch.float64)
        gap = torch.tensor([-2.0, 1.5, 4.0], dtype=torch.float64)
        width = 2.0 ** torch.tensor([-8.0, -11.0, -14.0], dtype=torch.float64)
        r, gap, width = (v.flatten() for v in torch.meshgrid(r, gap, width, indexing="ij"))
        s = r + gap
        d = width * r.abs().clamp(min=1)
        linear = torch.stack([torch.ones_like(r), -(r + s), r * s], -1)
        pair = torch.stack([torch.ones_like(r), -2 * r, r * r + d * d], -1)
        coeffs = torch.zeros(len(r), 5, dtype=torch.float64)
        for i in range(3):
            for j in range(3):
                coeffs[:, i + j] += linear[:, i] * pair[:, j]
        expected = torch.stack([torch.maximum(r, s), torch.minimum(r, s)], -1)
        expected = torch.cat([expected, torch.zeros_like(expected)], -1)
        actual = solver.solve_quartic(coeffs.to(device=device, dtype=dtype))
        self.assert_close(actual, expected.to(device=device, dtype=dtype), atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize(
        "coefficients, expected, dtypes",
        [
            # Four real roots within 2% of each other: the resolvent has a near-triple root.
            (
                [1.0, -8.8321223404358, 29.251986018365162, -43.0583174138504, 23.767524638771818],
                [2.2279948162159041, 2.2083975939262021, 2.2081449386622555, 2.1875849916314388],
                (torch.float64,),
            ),
            # Two real roots and a complex pair in one cluster.
            (
                [1.0, -0.7960158586502075, 0.23761533200740814, -0.03152422606945038, 0.0015683587407693267],
                [0.20183169083788146, 0.19617540615477025],
                (torch.float32, torch.float64),
            ),
            # A separated real pair inside a cluster with a complex pair.
            (
                [-0.0014945328030236183, 0.13165948745769898, -4.349402922389792, 63.8592944813424, -351.6004102965555],
                [22.119540865055652, 21.998521966226848],
                (torch.float64,),
            ),
            # A real root beside a close complex pair, and a far root.
            (
                [0.28430994261445064, -1.8145339507785327, 3.534044520265233, -2.7811262287698018, 0.7766973115733834],
                [3.6612115102922415, 0.90699103508695153],
                (torch.float64,),
            ),
            (
                [1.0, -4.024720362857346, 5.810396766987136, -3.469536959883321, 0.6775864610358943],
                [1.215961780224748, 0.37690253008267075],
                (torch.float64,),
            ),
            # A near-double complex pair that Ferrari split between two real factors.
            (
                [-0.7672792631437675, -0.2277980718973569, 2.8115738175174863, 0.3210495206426153, 0.00913104309961989],
                [1.8310604836611197, -2.014346817978239],
                (torch.float64,),
            ),
            # A near-triple root with one real root: a second copy of it is not a root.
            (
                [1.0, -2.01993465423584, -0.6778868906849311, -0.07099884823303, -0.0024409612243582344],
                [2.3248481750488281, -0.10163741311517308],
                (torch.float64,),
            ),
            # A near-quadruple root with no real roots.
            (
                [-0.010004346039634609, -0.01847789435683093, -0.01279815961393944, -0.003939671824814085, -0.00045478259830212733],
                [],
                (torch.float64,),
            ),
        ],
    )
    def test_real_roots_in_clusters(self, coefficients, expected, dtypes, device, dtype):
        if dtype not in dtypes:
            pytest.skip("The coefficients are exact in the listed dtypes only.")
        # References: sympy real-root isolation of the represented coefficients. Inside a cluster a root is
        # determined to about eps^(1/k) for a k-fold cluster, hence the tolerance.
        coeffs = torch.tensor([coefficients], device=device, dtype=dtype)
        expected = torch.tensor([expected + [0.0] * (4 - len(expected))], device=device, dtype=dtype)
        actual = solver.solve_quartic(coeffs)
        assert int((actual != 0).sum()) == int((expected != 0).sum())
        self.assert_close(actual, expected, atol=0.0, rtol=1e-5)

    @pytest.mark.parametrize(
        "coefficients, expected, literal_dtype, rtol",
        [
            # Three close roots and a far one: their stationary points are classified only inside the local
            # quadratic region (cubic and quartic terms at most an eighth of the curvature term).
            (
                [1.0, 7.1414729471261325, 11.216197417652605, -17.58471566143806, -41.87878316906568],
                [1.6589838853480642, -2.9334722967452928, -2.9334786391423401, -2.9335058965865637],
                "float64",
                1e-6,
            ),
            # Two small roots beside a large complex pair: not a cluster. Centring rows whose centred root bound
            # reaches half the root bound loses the small root.
            (
                [1.0, 159457.2390277729, 6356652658.430805, -8862577.852632554, -15.019852486487155],
                [0.0013959135242863893, -1.6926947842228866e-6],
                "float64",
                1e-9,
            ),
            (
                [1.0, -40172.53538678672, 403458261.3150447, -2221380.9118629876, 3040.3600233223606],
                [0.0029599318540459437, 0.0025459210183159004],
                "float64",
                1e-9,
            ),
            # A near-triple root beside a fourth one: centring it needs a gate of a quarter, not a sixteenth.
            (
                [1.0, -6.542885780334473, 15.241097447951688, -15.246880367553306, 5.582061569668052],
                [2.7396306991577154, 1.2677443459345709],
                "float64",
                1e-5,
            ),
            # (x - 2)^3 (x + 9) with b one ulp up: one real root at 2. A resolvent root counts as dominant only
            # 16 times the others' size away; at twice, the closed form's triple comes back.
            ([1.0, 3.0000000000000004, -42.0, 100.0, -72.0], [1.9999931389943847, -9.0000000000000002], "float64", 1e-5),
            # An exact double root at 1 inside a near-quadruple cluster. A stationary point's value certifies the
            # critical value only beyond the drift p'^2 / |p''| of the point's own error; without it this
            # minimum reads as positive and both roots are lost.
            (
                [1.0, -4.008663177490234, 6.026017665863037, -4.026045799255371, 1.0086913108825684],
                [1.0, 1.0],
                "float32",
                1e-6,
            ),
            # No real roots: without the final residual test the pair at 0.486 is reported.
            (
                [0.24079275675471995, -0.5063849033527095, 0.3993460088867061, -0.13997015029289017, 0.0183972443730235],
                [],
                "float64",
                0.0,
            ),
            (
                [1.0, 2685.27490234375, -481271392.0, -13682.7939453125, -0.11872430890798569],
                [20636.308565148697, -23321.583439061931],
                "float32",
                1e-6,
            ),
        ],
    )
    def test_threshold_margins(self, coefficients, expected, literal_dtype, rtol, device, dtype):
        if dtype != getattr(torch, literal_dtype):
            pytest.skip("The coefficients are exact in the listed dtype; the row pins a threshold there.")
        # References: sympy real-root isolation of the represented coefficients. Each row fails when its
        # threshold is moved by a factor of 2 to 16 in the direction the comment names.
        coeffs = torch.tensor([coefficients], device=device, dtype=dtype)
        expected = torch.tensor([expected + [0.0] * (4 - len(expected))], device=device, dtype=dtype)
        actual = solver.solve_quartic(coeffs)
        assert int((actual != 0).sum()) == int((expected != 0).sum())
        self.assert_close(actual, expected, atol=0.0, rtol=rtol)

    def test_coefficient_underflowed_by_rescaling_counts_as_zero(self, device, dtype):
        if dtype != torch.float64:
            pytest.skip("The constant term is the smallest float64 subnormal.")
        # The documented limit: rescaled to a unit root bound, 5e-324 underflows, and the real pair
        # +-2.5e-163 is reported as a double root at 0.
        coeffs = torch.tensor([[1.0, 1.75, -78.0, 0.0, 5e-324]], device=device, dtype=dtype)
        expected = torch.tensor([[8.0, 0.0, 0.0, -9.75]], device=device, dtype=dtype)
        self.assert_close(solver.solve_quartic(coeffs), expected, atol=0.0, rtol=0.0)

    def test_exact_zero_double_root_gradient(self, device, dtype):
        if dtype != torch.float64:
            pytest.skip("The analytic Jacobian is compared in float64.")
        coeffs = torch.tensor([[1.0, -6.25, -1.625, 0.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        roots = solver.solve_quartic(coeffs)
        # The simple root 6.5 has the implicit Jacobian -[r^4, r^3, r^2, r, 1] / p'(r), e included.
        r = 6.5
        slope = 4 * r**3 - 18.75 * r**2 - 3.25 * r
        expected = -torch.tensor([[r**4, r**3, r**2, r, 1.0]], device=device, dtype=dtype) / slope
        (gradient,) = torch.autograd.grad(roots[0, 0], coeffs, retain_graph=True)
        self.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)
        # The double zero keeps the repeated-root convention: the four roots sum to -b / a.
        (gradient,) = torch.autograd.grad(roots.sum(), coeffs)
        expected = torch.tensor([[-6.25, -1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)

    @pytest.mark.parametrize(
        "coefficients, expected",
        [
            # Two distinct real roots near a stationary point. References below
            # solve the represented coefficients, not their generating roots.
            (
                [1.0, -52.14418276628962, 1007.5187742251757, -8555.967241283795, 26969.233894919536],
                [16.7821050275422, 13.7045941923794, 10.82874356, 10.82873999],
            ),
            # The corresponding near-axis complex pair must stay padding.
            (
                [1.0, -64.8997252545787, 1572.8087859375592, -16876.08468535899, 67670.40367585233],
                [19.322148600106, 14.574549379298, 0.0, 0.0],
            ),
            # Exact binary-float references: sympy.nroots with coefficients
            # constructed as sympy.Rational(float_value), n=50, maxsteps=1000.
            (
                [1.0, -64.94582150986697, 1579.9948574760515, -17065.022602512025, 69044.1926650335],
                [17.55969231808344, 16.66924737875942, 15.35847018582340, 15.35841162720071],
            ),
            (
                [1.0, -63.03908361158486, 1480.402492442505, -15345.546348060858, 59217.6023118378],
                [18.78915262046432, 12.52836237857898, 0.0, 0.0],
            ),
            (
                [1.0, 58.93510754073745, 1301.9503673298948, 12777.476117348911, 47004.325578865675],
                [-13.99561747187950, -14.47456758390348, -15.23238644901737, -15.23253603593710],
            ),
            # A nearly repeated resolvent can produce a completely invalid real
            # factor even though the quartic has no real roots (#4474).
            ([1.0, 14.0, 90.50379432823472, 290.5265602976431, 430.6412357523629], [0.0, 0.0, 0.0, 0.0]),
            # Four separated root magnitudes require stable factors and enough
            # polishing to converge, rather than accepting or dropping partial steps.
            ([1.0, 499999.99001, -500000004995.0, 4995000000.00000095, 50000.0], [500000.0, 0.01, -1e-5, -1000000.0]),
        ],
    )
    def test_resolvent_conditioning_boundaries(self, coefficients, expected, device, dtype):
        if dtype != torch.float64:
            pytest.skip("These literals separate float64 coefficient rounding from solver error.")
        # Generation/reference: numpy.roots(numpy.array(coefficients, dtype=numpy.float64)).
        coeffs = torch.tensor([coefficients], device=device, dtype=dtype)
        roots = solver.solve_quartic(coeffs)
        self.assert_close(roots, torch.tensor([expected], device=device, dtype=dtype), atol=3e-7, rtol=2e-11)

    @pytest.mark.parametrize(
        "coefficients",
        [
            [1.0, -64.94582150986697, 1579.9948574760515, -17065.022602512025, 69044.1926650335],
            [1.0, 58.93510754073745, 1301.9503673298948, 12777.476117348911, 47004.325578865675],
        ],
    )
    def test_close_distinct_roots_preserve_vieta_gradient(self, coefficients, device, dtype):
        if dtype != torch.float64:
            pytest.skip("These close distinct roots require float64 input coefficients.")
        values = torch.tensor([coefficients], device=device, dtype=dtype, requires_grad=True)
        actual = solver.solve_quartic(values)
        assert bool((actual != 0).all())
        (gradient,) = torch.autograd.grad(actual.sum(), values)
        expected = torch.tensor([[coefficients[1], -1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(gradient, expected, atol=1e-8, rtol=1e-10)

    def test_close_pair_preserves_separated_root_jacobians(self, device, dtype):
        if dtype != torch.float64:
            pytest.skip("These analytic Jacobians require float64 input coefficients.")
        values = torch.tensor(
            [[1.0, -64.94582150986697, 1579.9948574760515, -17065.022602512025, 69044.1926650335]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        roots = solver.solve_quartic(values)
        # Exact-binary-float roots from sympy.nroots(..., n=50): implicit
        # derivative -[r**4, r**3, r**2, r, 1] / p'(r), independent of the solver.
        expected = torch.tensor(
            [
                [-22035.41543636148, -1254.8861926053694, -71.46401940727937, -4.069776287235045, -0.23176808645125746],
                [50463.77589809011, 3027.3577895540107, 181.6133458677674, 10.895113722961836, 0.6536056173024822],
            ],
            device=device,
            dtype=dtype,
        )
        for slot in range(2):
            (gradient,) = torch.autograd.grad(roots[0, slot], values, retain_graph=True)
            self.assert_close(gradient[0], expected[slot], atol=1e-8, rtol=1e-7)

    def test_conditioning_dynamo(self, device, dtype, torch_optimizer, optimizer_backend):
        if optimizer_backend == "jit" or device.type != "cpu" or dtype != torch.float64:
            pytest.skip("Fullgraph conditioning coverage requires CPU float64.")
        values = torch.tensor(
            [
                [1.0, -64.94582150986697, 1579.9948574760515, -17065.022602512025, 69044.1926650335],
                [1.0, -63.03908361158486, 1480.402492442505, -15345.546348060858, 59217.6023118378],
            ],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        expected = solver.solve_quartic(values)
        actual = torch_optimizer(solver.solve_quartic, fullgraph=True)(values)
        self.assert_close(actual, expected, atol=3e-7, rtol=2e-11)
        (gradient,) = torch.autograd.grad(actual[0].sum(), values)
        expected_gradient = torch.tensor(
            [[values[0, 1].item(), -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype
        )
        self.assert_close(gradient, expected_gradient, atol=1e-8, rtol=1e-10)

    def test_fullgraph_dynamo_mixed_degrees_and_backward(self, device, dtype, torch_optimizer, optimizer_backend):
        if optimizer_backend == "jit" or device.type != "cpu" or dtype not in (torch.float32, torch.float64):
            pytest.skip("Fullgraph Inductor coverage uses CPU float32/float64.")
        coeffs = torch.tensor(
            [
                [1.0, -10.0, 35.0, -50.0, 24.0],
                [1.0, 9.5, 33.25, 50.625, 28.125],
                [1.0, 0.0, 0.0, 0.0, -16.0],
                [1.0, 4.0, 14.01, 20.02, 25.05],
                [0.0, 1.0, -6.0, 11.0, -6.0],
                [0.0, 0.0, 0.0, 2.0, -6.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        compiled = torch_optimizer(solver.solve_quartic, fullgraph=True)
        actual = compiled(coeffs)
        expected = solver.solve_quartic(coeffs)
        self.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        grad = torch.autograd.grad(actual.sum(), coeffs, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.sum(), coeffs)[0]
        assert grad.isfinite().all()
        self.assert_close(grad, expected_grad, atol=1e-4, rtol=1e-4)

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
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip(
                "Half coefficient rounding changes close-root multiplicity; exact-input cases cover these dtypes."
            )
        # Generate random roots and construct coefficients to ensure valid solutions exist
        torch.manual_seed(0)
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

    def test_close_real_roots_bound_4906(self, device, dtype):
        # #4906: row 4 of torch.manual_seed(95); torch.randn(10, 4) on CPU, through test_random's float32
        # coefficient construction. Two real roots sit 0.056 apart. The float32 rounding of the coefficients
        # alone moves the roots by up to 4.1e-4 (the float64 solve of these coefficients). Keep the literals
        # exact: moving two coefficients by one ulp removes the original 2.3e-2 float32 solver-side error.
        # Both dtypes are now bounded by coefficient rounding, separately from solver accuracy below.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("the case bounds the float32 loss and the float64 coefficient rounding")
        coeffs = torch.tensor(
            [[1.0, 4.4621620178222656, 7.4423127174377441, 5.4986057281494141, 1.518401026725769]],
            device=device,
            dtype=dtype,
        )
        true_roots = torch.tensor(
            [[-1.2491413354873657, -1.1928491592407227, -1.0452626943588257, -0.974908709526062]],
            device=device,
            dtype=dtype,
        )
        out = solver.solve_quartic(coeffs).sort(-1).values
        self.assert_close(out, true_roots, atol=1e-3, rtol=0.0)

    def test_close_real_roots_same_coefficients_4906(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("this regression separates float32 solver error from coefficient rounding")
        # Exact float32 coefficients: issue seeds 95 and 10, a cluster for which float32 lost all
        # candidates, and a well-separated control. References solve THESE coefficients, not the
        # generating roots: numpy.roots(np.array(row, dtype=np.float64)), cross-checked with float64
        # solve_quartic. In particular seed 10's rounded coefficients widen its pair gap to 0.00831.
        coeffs = torch.tensor(
            [
                [1.0, 4.4621620178222656, 7.4423127174377441, 5.4986057281494141, 1.518401026725769],
                [1.0, -4.067328453063965, 6.175639629364014, -4.146894454956055, 1.0385831594467163],
                [1.0, 8.073843002319336, 24.425722122192383, 32.81570816040039, 16.51930809020996],
                [1.0, -10.0, 35.0, -50.0, 24.0],
            ],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor(
            [
                [-1.2488913493704268, -1.193254868897449, -1.0449774473549063, -0.9750383521990926],
                [0.8297586315359857, 1.0000497091697647, 1.114603594165587, 1.1229165181917928],
                [-2.126456753929611, -2.0938150726449853, -1.9748027394050875, -1.8787684363396553],
                [1.0, 2.0, 3.0, 4.0],
            ],
            device=device,
            dtype=dtype,
        )
        roots = solver.solve_quartic(coeffs).sort(dim=-1).values
        assert bool((roots != 0).all()), "all four roots must survive, including seed 10's close pair"
        # Casting the high-precision answer costs at most half an ulp. Float64 leaves a little room
        # for Ferrari's cancellation on the clustered rows and for the independent reference solve.
        rtol = 2 * torch.finfo(dtype).eps if dtype == torch.float32 else 1e-8
        self.assert_close(roots, expected, atol=0.0, rtol=rtol)
        residual = coeffs[:, :1].expand_as(roots)
        scale = residual.abs()
        for column in range(1, 5):
            residual = residual * roots + coeffs[:, column : column + 1]
            scale = scale * roots.abs() + coeffs[:, column : column + 1].abs()
        self.assert_close(residual / scale, torch.zeros_like(residual), atol=8 * torch.finfo(dtype).eps, rtol=0.0)
        for row in range(len(coeffs)):
            self.assert_close(solver.solve_quartic(coeffs[row : row + 1]).sort(dim=-1).values, roots[row : row + 1])

    def test_close_real_roots_float32_gradient_4906(self, device):
        coeffs = torch.tensor(
            [[1.0, -4.067328453063965, 6.175639629364014, -4.146894454956055, 1.0385831594467163]],
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        # Check the gradient through the precision retry against the same coefficients in float64.
        # MPS follows the documented CPU fallback, including the copies in both directions.
        reference = coeffs.detach().cpu().double().requires_grad_()
        weights = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device)
        (solver.solve_quartic(coeffs).sort(dim=-1).values * weights).sum().backward()
        (solver.solve_quartic(reference).sort(dim=-1).values * weights.cpu().double()).sum().backward()
        assert bool(torch.isfinite(coeffs.grad).all())
        self.assert_close(coeffs.grad, reference.grad.to(coeffs), atol=0.0, rtol=2 * torch.finfo(torch.float32).eps)

    def test_close_real_roots_retry_margin_4906(self, device):
        # The retry's rounding estimate needs its margin. These float32 rows have |discriminant| at 0.37 and 0.26
        # of the estimated uncertainty. With a quarter of the eight-epsilon roundoff they are no longer retried:
        # the first then reports its complex pair -0.4795 +- 6.0e-4j as two real roots, the second loses its real
        # pair 1.2417, 1.2548 (gap 0.013) to zero padding. Real roots: numpy.roots of the exact float32 values.
        coeffs = torch.tensor(
            [
                [1.0, 1.684098243713379, 0.8085256814956665, 0.05472347512841225, -0.02685355208814144],
                [1.0, -4.816625595092773, 9.516379356384277, -9.02255916595459, 3.3748888969421387],
            ],
            device=device,
            dtype=torch.float32,
        )
        expected = torch.tensor([[-0.8607131, 0.13567771], [1.24165737, 1.2547649]], device=device, dtype=torch.float32)
        roots = solver.solve_quartic(coeffs)
        assert (roots != 0).sum(-1).tolist() == [2, 2], f"expected two real roots per row, got {roots.tolist()}"
        found = roots[roots != 0].view(2, 2).sort(dim=-1).values
        self.assert_close(found, expected, atol=0.0, rtol=1e-5)

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
            # Historical regression root sets. Test the input polynomial and its multiplicities,
            # independently of the solver's candidate construction and acceptance implementation.
            ([1.0, 15.49, 58.495, 144.38, 1300.075], [-8.75, -8.74], (torch.float32, torch.float64)),
            # A small root survives next to roots of 1e2 to 1e3.
            (
                [1.0, -1088.503, -128841.7345, 329286.535, -986.7],
                [-110.0, 0.003, 2.5, 1196.0],
                (torch.float32, torch.float64),
            ),
            # Float64 coefficient rounding turns the generating double root at 2.3 into a complex pair
            # (numpy.roots of these coefficients: 2.3 +/- 2.71e-8j). Preserve the two simple roots.
            (
                _monic_from_roots(torch.tensor([[2.3, 2.3, -4.19, 1.68]], dtype=torch.float64))[0].tolist(),
                [-4.19, 1.68],
                (torch.float64,),
            ),
            # Only two real roots: -2, -1.5; the other pair is 4 +/- 0.5i.
            ([1.0, -4.5, -8.75, 32.875, 48.75], [-2.0, -1.5], (torch.float32, torch.float64)),
            # Two real roots beside a large complex pair.
            (
                [1.0, 8258.0, 15691705.0, -1590869938.0, 15479948400000.0],
                [-4689.0, -4031.0],
                (torch.float32, torch.float64),
            ),
            ([1.0, 12.0, 27.0, 50.0, 450.0], [-9.0, -5.0], (torch.float32, torch.float64)),
            # Preserve both copies of the exact double root at -2.
            ([1.0, 6.75, 3.75, -34.0, -45.0], [-5.0, -2.0, -2.0, 2.25], (torch.float32, torch.float64)),
            # Exactly two real roots, 4.75 and -5, beside 9 +/- 3i.
            ([1.0, -17.75, 61.75, 450.0, -2137.5], [-5.0, 4.75], (torch.float32, torch.float64)),
            # Two complex pairs, no real roots.
            ([1.0, 27.91975997, 297.7351036, 1435.935501, 2645.836994], [], (torch.float32,)),
            # This originally pinned an approximate double root at 0.558935. The actual float32
            # coefficients have a complex pair 0.5589346 +/- 1.17188435e-5j (numpy.roots in float64).
            # The precision retry must reject that pair rather than preserve two spurious real roots.
            (
                [1.0, -7.636022673, 0.9114557167, 5.439310611, -2.089195035],
                [-0.901329, 7.419483],
                (torch.float32,),
            ),
            # A small root beside roots of 1, 386 and 946.
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

    def test_residual_tolerance_for_half_inputs_4474(self, device, dtype):
        # The represented float16 coefficients in each row have exactly two
        # real roots. Classification must preserve the close complex pair as padding.
        if dtype != torch.float16:
            pytest.skip("Half inputs are where the float32 residual tolerance still applies.")
        coeffs = torch.tensor(
            [
                [1.0, -4.8203125, 8.6640625, -6.8828125, 2.033203125],
                [1.0, 0.57568359375, -0.11761474609375, -0.1134033203125, -0.01605224609375],
            ],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor([[0.88545993, 1.53287539], [-0.46115802, 0.44471336]], device=device, dtype=dtype)
        roots = solver.solve_quartic(coeffs)
        assert (roots != 0).sum(-1).tolist() == [2, 2], f"expected two real roots per row, got {roots.tolist()}"
        found = roots[roots != 0].view(2, 2).sort(dim=-1).values
        self.assert_close(found, expected, atol=0.0, rtol=1e-2)

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

    def test_small_scale_quartic_literal_4833(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("this row's 2e-11 and -4.2e-08 coefficients underflow float16 and keep 3 digits in bfloat16")
        # #4833: x^4 - 0.009x^3 + 3e-05x^2 - 4.2e-08x + 2e-11 = (x - 0.001)(x - 0.002)(x^2 - 0.006x + 1e-05) has
        # the real roots 1e-3 and 2e-3. This literal lost them in the resolvent cubic, whose R = 2e-18 sat under
        # the 1e-16 floor removed with #4914; the additional scale families below pin the quartic's own thresholds.
        coeffs = torch.tensor([[1.0, -0.009, 3e-05, -4.2e-08, 2e-11]], device=device, dtype=dtype)
        out = solver.solve_quartic(coeffs)
        for root in (1e-3, 2e-3):
            assert (out - root).abs().min() <= 1e-5 * root, out

    @pytest.mark.parametrize("exponent", [-4, -8, -16, -24, -32])
    def test_small_scale_quartic_preserves_four_real_roots_4833(self, exponent, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("The scaled coefficients are not representable accurately in half precision.")
        # Replacing x by 2**exponent * u in (u-1)(u-2)(u-3)(u-4) scales coefficient k by 2**(k*exponent).
        # All four roots stay real and separated; unit-floored Ferrari thresholds lost two at small scales.
        scale = 2.0**exponent
        # Form the final values before transfer: at exponent -32, float32 scale**4 is subnormal,
        # while 24 * scale**4 is normal. Metal may flush that intermediate power to zero.
        coefficients = torch.tensor(
            [[value * scale**power for power, value in enumerate((1.0, -10.0, 35.0, -50.0, 24.0))]],
            device=device,
            dtype=dtype,
        )
        roots = solver.solve_quartic(coefficients).sort(dim=-1).values / scale
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device, dtype=dtype)
        self.assert_close(roots, expected, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("exponent", [-4, -8, -16, -32])
    def test_small_scale_quartic_does_not_create_real_roots_4833(self, exponent, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("The scaled coefficients are not representable accurately in half precision.")
        # (u^2 + 2u + 5)(u^2 + 2u + 5.01) has two negative quadratic discriminants, at every scale.
        scale = 2.0**exponent
        # Avoid the subnormal scale**4 intermediate while preserving the intended normal coefficients.
        coefficients = torch.tensor(
            [[value * scale**power for power, value in enumerate((1.0, 4.0, 14.01, 20.02, 25.05))]],
            device=device,
            dtype=dtype,
        )
        roots = solver.solve_quartic(coefficients)
        self.assert_close(roots, torch.zeros_like(roots), rtol=0, atol=0)

    def test_small_scale_quartic_gradcheck_4833(self, device):
        # Perturb the unit-scale coefficients, then change variables. Perturbing tiny physical coefficients
        # by gradcheck's absolute epsilon would change the root family instead of checking its local Jacobian.
        coefficients = torch.tensor([[1.0, -10.0, 35.0, -50.0, 24.0]], device=device, dtype=torch.float64)
        powers = torch.arange(5, device=device, dtype=torch.float64)
        scale = 2.0**-16
        coefficient_scale = scale**powers

        def scaled_solver(value):
            return solver.solve_quartic(value * coefficient_scale) / scale

        self.gradcheck(scaled_solver, (coefficients.requires_grad_(),))

    def test_convention_solve_quartic_scale_invariant_large_roots_4954(self, device, dtype):
        if dtype != torch.float32:
            pytest.skip(
                "pinned in float32, where this row's 6e-8 ratio of leading to largest coefficient is below 1e-6"
            )
        # #4954: (x - 50)(x - 60)(x - 70)(x - 80) times 2^-21 has a = 4.8e-7, below 1e-6 (its largest coefficient
        # is 8.01, so the old test is absolute), but its root bound is 260, so it stays a quartic. Today's error is
        # at most 1.5e-3; a cubic fallback misses by at least 18.
        row = torch.tensor([[1.0, -260.0, 25100.0, -1066000.0, 16800000.0]], device=device, dtype=dtype)
        out = solver.solve_quartic(row * 2.0**-21).sort(dim=-1).values
        expected = torch.tensor([[50.0, 60.0, 70.0, 80.0]], device=device, dtype=dtype)
        self.assert_close(out, expected, atol=1e-2, rtol=0.0)

    def test_convention_solve_quartic_scale_invariant_large_roots_float64_4954(self, device, dtype):
        if dtype != torch.float64:
            pytest.skip("the float32 case is pinned in test_convention_solve_quartic_scale_invariant_large_roots_4954")
        # Same shape as the float32 pin, at float64's 1e-12 tolerance: (x - 1e4)...(x - 4e4) scaled by 2^-50 has
        # a = 8.9e-16, below 1e-12 (largest coefficient 213, so the old test is absolute), but its root bound is
        # 1e5, so it stays a quartic. Today's error is at most 4.4e-11; a cubic fallback misses by at least 946.
        row = torch.tensor([[1.0, -100000.0, 3500000000.0, -50000000000000.0, 2.4e17]], device=device, dtype=dtype)
        out = solver.solve_quartic(row * 2.0**-50).sort(dim=-1).values
        expected = torch.tensor([[10000.0, 20000.0, 30000.0, 40000.0]], device=device, dtype=dtype)
        self.assert_close(out, expected, atol=1e-6, rtol=1e-9)

    def test_quartic_keeps_small_real_roots_next_to_large_complex_pair_5348(self, device, dtype):
        if dtype != torch.float32:
            pytest.skip("this issue is specific to float32 resolvent accuracy")
        # (x - 0.5)(x + 0.25)(x^2 + M): the real roots stay fixed while the other pair grows.
        values = [1e5, 1e6, 1e7, 1e8]
        coeffs = torch.tensor(
            [[1.0, -0.25, value - 0.125, -0.25 * value, -0.125 * value] for value in values],
            device=device,
            dtype=dtype,
        )

        roots = solver.solve_quartic(coeffs)

        assert torch.equal((roots != 0).sum(dim=-1), torch.full((len(values),), 2, device=device))
        expected = torch.tensor([[-0.25, 0.0, 0.0, 0.5]] * len(values), device=device, dtype=dtype)
        self.assert_close(roots.sort(dim=-1).values, expected, atol=1e-4, rtol=1e-4)

    def test_quartic_keeps_small_real_roots_next_to_large_real_pair_5348(self, device, dtype):
        if dtype != torch.float32:
            pytest.skip("the cancellation pinned here is a float32 one")
        # (x - 0.5)(x + 0.25)(x - 1e4)(x - 2e4): the large pair is real, so recovering the small roots from zero
        # placeholders afterwards cannot reach them; rebuilding the small factor from the large one does.
        coeffs = torch.tensor([[1.0, -30000.25, 200007499.875, -49996250.0, -25000000.0]], device=device, dtype=dtype)

        roots = solver.solve_quartic(coeffs).sort(dim=-1).values

        expected = torch.tensor([[-0.25, 0.5, 10000.0, 20000.0]], device=device, dtype=dtype)
        self.assert_close(roots, expected, atol=1e-5, rtol=1e-6)

    def test_quartic_close_small_pair_next_to_large_complex_pair_float64_5348(self, device, dtype):
        if dtype != torch.float64:
            pytest.skip("float64 rows; the float32 family is pinned above")
        # (x - 0.01)(x - 0.0105)(x^2 + 2000 x + 4e9) and (x + 0.03)(x + 0.0315)(x^2 + 100 x + 1e10). The cancellation
        # also costs float64: the first row lost both real roots and the second was 1.9e-7 off. The complex pair's
        # nonzero linear term makes the small factor's x coefficient depend on b_large * c_small.
        coeffs = torch.tensor(
            [
                [1.0, 1999.9795, 3999999959.000105, -81999999.79, 420000.0],
                [1.0, 100.0615, 10000000006.150944, 615000000.0945, 9450000.0],
            ],
            device=device,
            dtype=dtype,
        )

        roots = solver.solve_quartic(coeffs)

        assert torch.equal((roots != 0).sum(dim=-1), torch.full((2,), 2, device=device))
        expected = torch.tensor([[0.0, 0.0, 0.01, 0.0105], [-0.0315, -0.03, 0.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(roots.sort(dim=-1).values, expected, atol=0.0, rtol=1e-12)

    def test_quartic_gradient_finite_with_tiny_quadratic_coefficient(self, device, dtype):
        # x^4 + 1e-12 x^2 + x + 1 has no real roots. A candidate derived from B x^2 + C x + D would sit near
        # -C / B = -1e12, whose fourth power overflows float32 in a discarded torch.where lane and turns every
        # coefficient gradient into nan, although the returned roots are all zero.
        coeffs = torch.tensor([[1.0, 0.0, 1e-12, 1.0, 1.0]], device=device, dtype=dtype, requires_grad=True)

        roots = solver.solve_quartic(coeffs)
        roots.sum().backward()

        assert torch.equal(roots.detach(), torch.zeros_like(roots))
        assert torch.isfinite(coeffs.grad).all()

    def test_convention_solve_quartic_tiny_leading_coefficient_real_roots_4954(self, device, dtype):
        if dtype != torch.float32:
            pytest.skip("pinned in float32, where the old absolute tolerance (1e-6) is what this row crosses")
        # #4954: x^4 - 1 times 1e-7 has a = 1e-7, below 1e-6, but its root bound is 56.23 (|e/a|^(1/4)), so it
        # stays a quartic, recovering the real roots +-56.23. Today's error is at most 3.8e-6; a cubic
        # fallback misses by 56.23 (both roots lost to [0, 0, 0, 0]).
        coeffs = torch.tensor([[1e-7, 0.0, 0.0, 0.0, -1.0]], device=device, dtype=dtype)
        out = solver.solve_quartic(coeffs)
        root = 1e7**0.25
        for expected_root in (-root, root):
            assert (out - expected_root).abs().min() <= 1e-4 * abs(expected_root)

    def test_convention_solve_quartic_and_rule_keeps_dominant_root_4954(self, device, dtype):
        if dtype != torch.float32:
            pytest.skip("pinned in float32; this row's unit-scale absolute test is what the AND rule must not override")
        # #4954: this row has a = 1e-5, above 1e-6, so the old test alone already keeps it a quartic, finding
        # its root near -1e7. Its root bound is 1e7 (b/a): the bound ALONE, not ANDed to the old test, would
        # wrongly fall back here. Today's error is at most 5.4e-7; a bound-alone regression loses the -1e7 root.
        coeffs = torch.tensor([[1e-5, 100.0, -50.0, 3.0, -1.0]], device=device, dtype=dtype)
        out = solver.solve_quartic(coeffs)
        assert (out - (-10000000.5)).abs().min() <= 10.0
        assert (out - 0.48086).abs().min() <= 1e-5 * 0.48086

    def test_convention_solve_quartic_roots_below_quarter_inverse_tolerance_any_scale_4954(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("the constant term of these rows overflows float16 and keeps 3 digits in bfloat16")
        # The root bound is at most 4 times the largest root, so roots all below 1 / (4 * tol) (2.5e5 in float32,
        # 2.5e11 in float64) keep a row on the quartic path at every scale, here down to a leading coefficient of
        # 8e-25. Each term of the bound crosses 1 / tol here if its power of tol is off by one, and the row then
        # falls back to the cubic, as the leading-coefficient test alone does once a drops below tol (#4954).
        unit = 1.0 if dtype == torch.float32 else 1e6
        roots = [-2.4e5 * unit, -1.5e5 * unit, 1e5 * unit, 2e5 * unit]
        monic = [1.0, 9e4 * unit, -6.1e10 * unit**2, -3e15 * unit**3, 7.2e20 * unit**4]
        coeffs = torch.tensor([[c * 2.0**-k for c in monic] for k in (0, 20, 40, 60, 80)], device=device, dtype=dtype)
        out = solver.solve_quartic(coeffs).sort(dim=-1).values
        expected = torch.tensor([roots] * 5, device=device, dtype=dtype)
        self.assert_close(out, expected, rtol=1e-4, atol=0.0)

    def test_convention_solve_quartic_relative_leading_tolerance_4905(self, device, dtype):
        # (x - 1)(x - 2)(x - 3)(x - 4), and the same row times a power of two (exact in every dtype) that brings the
        # leading coefficient below the fallback tolerance (1e-6, or 1e-12 in float64).
        row = torch.tensor([[1.0, -10.0, 35.0, -50.0, 24.0]], device=device, dtype=dtype)
        scale = 2.0**-41 if dtype == torch.float64 else 2.0**-21
        roots = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device, dtype=dtype)
        self.assert_close(solver.solve_quartic(row).sort(dim=-1).values, roots)
        # #4905: below unit scale the tolerance is relative to the row's largest coefficient, so scaling the row
        # down does not turn it into a cubic.
        self.assert_close(solver.solve_quartic(row * scale).sort(dim=-1).values, roots)

    def test_convention_solve_quartic_cubic_fallback_4905(self, device, dtype):
        # Rows at unit scale or above keep the absolute tolerance: (x - 50)(x - 60)(x - 70)(x - 80) has a leading
        # coefficient 6e-8 times its constant term and is still a quartic, while a leading coefficient below the
        # tolerance on a unit-scale row, or an exact 0 on an all-zero row, still falls back to the cubic. Below unit
        # scale the test is relative in both directions: the unit-scale fallback row times 2^-14 falls back as well.
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the 50..80 row's 1.68e7 constant term overflows float16 and keeps 3 digits in bfloat16")
        tiny = 1e-13 if dtype == torch.float64 else 1e-7
        cubic_row = [tiny, 1.0, -6.0, 11.0, -6.0]
        coeffs = torch.tensor(
            [
                [1.0, -260.0, 25100.0, -1066000.0, 16800000.0],
                cubic_row,
                [c * 2.0**-14 for c in cubic_row],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        expected = torch.tensor(
            [[50.0, 60.0, 70.0, 80.0], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        roots = solver.solve_quartic(coeffs)
        self.assert_close(roots.detach().sort(dim=-1).values, expected, atol=1e-3, rtol=1e-4)
        # The all-zero row's relative tolerance is 0; the exact a == 0 test keeps it off the quartic path, where its
        # gradient is nan.
        (grad,) = torch.autograd.grad(roots.sum(), coeffs)
        assert grad.isfinite().all()


class TestSolveCubicReal(BaseTester):
    """The private seven-point cubic kernel: real roots, a validity mask, and gradients from the Newton step."""

    def _skip_half(self, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the kernel runs in the seven-point solvers' float32/float64 solve dtype")

    def test_repeated_root_newton_step(self, device, dtype):
        self._skip_half(dtype)
        coeffs = torch.tensor([[1.0, 0.0, -0.75, 0.25]], device=device, dtype=dtype, requires_grad=True)
        roots, valid = _solve_cubic_real(coeffs)
        assert valid.all()
        expected = torch.tensor([[-1.0, 0.5, 0.5]], device=device, dtype=dtype)
        self.assert_close(roots.sort(dim=1).values, expected)
        roots.sum().backward()
        assert torch.isfinite(coeffs.grad).all()

    def test_three_real_roots(self, device, dtype):
        self._skip_half(dtype)
        # (x - 1)(x - 2)(x - 3)
        coeffs = torch.tensor([[1.0, -6.0, 11.0, -6.0]], device=device, dtype=dtype)
        roots, valid = _solve_cubic_real(coeffs)
        assert valid.all()
        self.assert_close(roots.sort(dim=1).values, torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype))

    def test_one_real_root_is_repeated_in_the_masked_slots(self, device, dtype):
        self._skip_half(dtype)
        # x^3 + x + 1 has one real root, -0.6823278...
        coeffs = torch.tensor([[1.0, 0.0, 1.0, 1.0]], device=device, dtype=dtype)
        roots, valid = _solve_cubic_real(coeffs)
        assert valid.tolist() == [[True, False, False]]
        assert torch.isfinite(roots).all()
        self.assert_close(roots, roots[:, :1].expand(-1, 3))
        self.assert_close(roots[:, 0], torch.tensor([-0.6823278038280193], device=device, dtype=dtype))

    @pytest.mark.parametrize(
        "coeffs", [[1.0, 0.0, 1.0, 1.0], [1.0, -6.0, 11.0, -6.0], [1.0, 0.0, 0.0, -8.0], [1.0, -3.0, 3.0, -1.0]]
    )
    def test_backward_is_finite(self, device, dtype, coeffs):
        # One real root, three, a vanishing depressed coefficient (x^3 = 8 takes a cube root of 0 in Cardano's formula)
        # and a triple root: the closed form's guards have unbounded derivatives at all of them (#4229).
        self._skip_half(dtype)
        c = torch.tensor([coeffs], device=device, dtype=dtype, requires_grad=True)
        roots, valid = _solve_cubic_real(c)
        (roots * valid).sum().backward()
        assert torch.isfinite(c.grad).all()

    def test_gradient_is_the_implicit_derivative(self, device, dtype):
        self._skip_half(dtype)
        # For a simple root, d root / d c_i = -x^(3 - i) / p'(x): x^3 - 8 has the root 2 and p'(2) = 12.
        c = torch.tensor([[1.0, 0.0, 0.0, -8.0]], device=device, dtype=dtype, requires_grad=True)
        roots, _ = _solve_cubic_real(c)
        roots[0, 0].backward()
        expected = -torch.tensor([[8.0, 4.0, 2.0, 1.0]], device=device, dtype=dtype) / 12.0
        self.assert_close(c.grad, expected)

    @pytest.mark.parametrize("coeffs", [[1.0, 0.0, 1.0, 1.0], [1.0, -6.0, 11.0, -6.0], [1.0, 0.0, 0.0, -8.0]])
    def test_gradcheck(self, device, coeffs):
        c = torch.tensor([coeffs], device=device, dtype=torch.float64)
        self.gradcheck(lambda c: _solve_cubic_real(c)[0], (c,))
