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

from kornia.geometry.solvers import solve_cubic, solve_quadratic
from kornia.geometry.solvers.polynomial_solver import _solve_cubic_with_count

from testing.base import BaseTester


class TestPolynomialSolversCompile(BaseTester):
    @pytest.mark.parametrize(
        "solve, coeffs",
        [
            (
                solve_quadratic,
                [[1.0, -5.0, 6.0], [1.0, 0.0, 1.0], [0.0, 2.0, -6.0], [0.0, 0.0, 3.0]],
            ),
            (
                solve_cubic,
                [
                    [1.0, -6.0, 11.0, -6.0],
                    [1.0, 0.0, 1.0, -2.0],
                    [1.0, -6.0, 9.0, -4.0],
                    [0.0, 1.0, -5.0, 6.0],
                    [0.0, 0.0, 2.0, -6.0],
                    [0.0, 0.0, 0.0, 3.0],
                ],
            ),
        ],
    )
    def test_dynamo_fullgraph_mixed_degrees(self, solve, coeffs, device, dtype, torch_optimizer, optimizer_backend):
        """Mixed real-root branches and degree lowering stay inside one Inductor graph."""
        if optimizer_backend == "jit" or device.type != "cpu" or dtype != torch.float32:
            pytest.skip("The fullgraph regression is explicitly exercised on CPU float32.")
        values = torch.tensor(coeffs, device=device, dtype=dtype, requires_grad=True)
        actual = torch_optimizer(solve, fullgraph=True)(values)
        expected = solve(values)
        self.assert_close(actual, expected)

        (gradient,) = torch.autograd.grad(actual.sum(), values)
        assert bool(torch.isfinite(gradient).all()), gradient

    @staticmethod
    def _exact_double_root_cubics(device, dtype):
        # (x - p)(x - q)^2 for every pair of distinct quarter-integers in [-8, 8]; the coefficients are exact.
        values = torch.arange(-32, 33, dtype=torch.float64) / 4
        p, q = (v.flatten() for v in torch.meshgrid(values, values, indexing="ij"))
        p, q = p[p != q], q[p != q]
        coeffs = torch.stack([torch.ones_like(p), -(p + 2 * q), q * q + 2 * p * q, -p * q * q], -1)
        exact = (coeffs.to(dtype).double() == coeffs).all(-1)
        roots = torch.stack([p, q, q], -1)[exact].sort(-1).values
        return coeffs[exact].to(device=device, dtype=dtype), roots.to(device=device, dtype=dtype)

    def test_dynamo_cuda_double_roots_without_contraction(self, device, dtype, torch_optimizer, optimizer_backend):
        if optimizer_backend != "inductor" or device.type != "cuda" or dtype not in (torch.float32, torch.float64):
            pytest.skip("The documented workaround concerns Inductor's Triton kernels on CUDA.")
        # The workaround named in solve_cubic's known-limitation note: without fused multiply-adds, the compensated
        # evaluation keeps every exact double root under Inductor on CUDA.
        coeffs, roots = self._exact_double_root_cubics(device, dtype)
        with torch._inductor.config.patch(emulate_precision_casts=True):
            actual = torch_optimizer(solve_cubic, fullgraph=True)(coeffs)
        self.assert_close(actual.sort(-1).values, roots, atol=1e-5, rtol=1e-5)

    @pytest.mark.xfail(
        strict=False,
        reason="Known limitation, not planned to be fixed: Triton's fused multiply-adds break the compensated "
        "evaluation under Inductor on CUDA, and 1-2 % of exact double roots come back as single roots.",
    )
    def test_wart_dynamo_cuda_contraction_loses_double_roots(self, device, dtype, torch_optimizer, optimizer_backend):
        if optimizer_backend != "inductor" or device.type != "cuda" or dtype not in (torch.float32, torch.float64):
            pytest.skip("The limitation concerns Inductor's Triton kernels on CUDA.")
        coeffs, roots = self._exact_double_root_cubics(device, dtype)
        actual = torch_optimizer(solve_cubic, fullgraph=True)(coeffs)
        self.assert_close(actual.sort(-1).values, roots, atol=1e-5, rtol=1e-5)

    def test_quadratic_extreme_scale_invariance(self, device, dtype):
        """Power-of-two homogeneous scaling preserves roots and gradients where b^2 and 4ac overflow."""
        if dtype == torch.float16:
            pytest.skip("float16 is solved in float32, where no float16 coefficient overflows b^2.")
        # Beyond the square root of the largest float, so that only the rescaling keeps the discriminant finite.
        exponent = {torch.bfloat16: 70, torch.float32: 70, torch.float64: 520}[dtype]
        base = torch.tensor([[1.0, -3.0, -4.0]], device=device, dtype=dtype, requires_grad=True)
        roots = solve_quadratic(base)
        (base_grad,) = torch.autograd.grad(roots.sum(), base)

        factor = torch.tensor(2.0**exponent, device=device, dtype=dtype)
        scaled = (base.detach() * factor).requires_grad_()
        scaled_roots = solve_quadratic(scaled)
        (scaled_grad,) = torch.autograd.grad(scaled_roots.sum(), scaled)
        self.assert_close(scaled_roots, roots.detach(), rtol=8 * torch.finfo(dtype).eps, atol=0.0)
        self.assert_close(scaled_grad * factor, base_grad, rtol=16 * torch.finfo(dtype).eps, atol=0.0)

    @pytest.mark.parametrize(
        "coeffs, expected",
        [
            ([1e38, 1.0, -1e-38], [6.18034e-39, -1.618034e-38]),
            ([1e-30, 0.0, -1e-30], [1.0, -1.0]),
        ],
    )
    def test_quadratic_extreme_terms_are_finite(self, coeffs, expected, device, dtype):
        if dtype != torch.float32:
            pytest.skip("The literal exercises float32 subnormal and overflow behavior.")
        values = torch.tensor([coeffs], device=device, dtype=dtype, requires_grad=True)
        roots = solve_quadratic(values)
        self.assert_close(roots, torch.tensor([expected], device=device, dtype=dtype), rtol=2e-5, atol=0.0)
        (gradient,) = torch.autograd.grad(roots.sum(), values)
        assert bool(torch.isfinite(gradient).all()), gradient

    def test_cubic_close_large_pair_uses_extended_precision(self, device, dtype):
        if dtype != torch.float32:
            pytest.skip("The literal exercises float32 discriminant cancellation.")
        values = torch.tensor(
            [[1.0, -481.0438232421875, 57850.8359375, -11.54153823852539]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        roots = solve_cubic(values)
        expected = torch.tensor([[240.56673, 0.000199505, 240.4769]], device=device, dtype=dtype)
        self.assert_close(roots.sort(dim=-1).values, expected.sort(dim=-1).values, rtol=2e-5, atol=1e-7)
        (gradient,) = torch.autograd.grad(roots.sum(), values)
        assert bool(torch.isfinite(gradient).all()), gradient

    def test_dynamo_close_pair_matches_eager(self, device, dtype, torch_optimizer, optimizer_backend):
        if optimizer_backend == "jit" or device.type != "cpu" or dtype != torch.float32:
            pytest.skip("The fullgraph regression is explicitly exercised on CPU float32.")
        # Three real roots 3.62100, 3.61973 and 2.41171, two of them close. A float32 per-row uncertainty bound on
        # the discriminant misses this row, so eager execution agrees with the graph only if it solves the row
        # in float64 as the graph does.
        values = torch.tensor(
            [[1.0, -9.652440071105957, 30.569580078125, -31.61037254333496]], device=device, dtype=dtype
        )
        eager = solve_cubic(values)
        compiled = torch_optimizer(solve_cubic, fullgraph=True)(values)
        self.assert_close(compiled, eager, rtol=0.0, atol=0.0)
        expected = torch.tensor([[2.41171, 3.61973, 3.62100]], device=device, dtype=dtype)
        self.assert_close(eager.sort(-1).values, expected, rtol=1e-5, atol=0.0)

    def test_cubic_lowered_quadratic_count_with_tiny_coefficients(self, device, dtype):
        if dtype != torch.float32:
            pytest.skip("The literal's products underflow in float32.")
        # 1e-30 (x^2 + x + 1) has no real roots, although every product in its discriminant underflows.
        _, count = _solve_cubic_with_count(torch.tensor([[0.0, 1e-30, 1e-30, 1e-30]], device=device, dtype=dtype))
        assert count.tolist() == [0]

    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_cubic_lowered_quadratic_count_uses_scaled_discriminant(self, sign, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("The literals exercise mixed-exponent float32/64 discriminants.")
        scale = sign * (1e308 if dtype == torch.float64 else 1e38)
        rows = torch.tensor([[0.0, scale, 0.0, abs(scale) ** -1]], device=device, dtype=dtype)
        _, count = _solve_cubic_with_count(rows)
        assert count.tolist() == ([0] if scale > 0 else [2])
