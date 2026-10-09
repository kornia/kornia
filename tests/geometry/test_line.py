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

import math

import pytest
import torch

from kornia.core.check import are_checks_enabled, disable_checks, enable_checks
from kornia.core.exceptions import TypeCheckError, ValueCheckError
from kornia.geometry.line import ParametrizedLine, fit_line
from kornia.geometry.plane import Hyperplane
from kornia.geometry.vector import Scalar, Vector3

from testing.base import BaseTester, assert_close


class TestParametrizedLine(BaseTester):
    def test_smoke(self, device, dtype):
        p0 = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        d0 = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        l0 = ParametrizedLine(p0, d0)
        self.assert_close(l0.origin, p0)
        self.assert_close(l0.direction, d0)
        assert l0.dim() == 2

    def test_through(self, device, dtype):
        p0 = torch.tensor([-1.0, -1.0], device=device, dtype=dtype)
        p1 = torch.tensor([1.0, 1.0], device=device, dtype=dtype)
        l1 = ParametrizedLine.through(p0, p1)
        direction_expected = torch.tensor([0.7071, 0.7071], device=device, dtype=dtype)
        self.assert_close(l1.origin, p0)
        self.assert_close(l1.direction, direction_expected)

    def test_through_coincident_points_direction_is_zero_5062(self, device, dtype):
        # #5062: through(p, p) normalizes the zero vector p1 - p0, which gave a NaN direction in float16 and zeros in
        # every other dtype. Coincident points are degenerate input that a value check may reject (#5041), so this
        # pins the arithmetic with the checks disabled: zeros in every dtype.
        p = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        p1 = p.clone().requires_grad_(True)
        checks_were_enabled = are_checks_enabled()
        disable_checks()
        try:
            line = ParametrizedLine.through(p, p1)
        finally:
            if checks_were_enabled:
                enable_checks()
        assert line.direction.dtype == dtype
        assert torch.equal(line.direction, torch.zeros(2, device=device, dtype=dtype))
        # The gradient at the zero direction is I / eps with eps = 1e-12, as with F.normalize, except in float16,
        # where I / eps overflows and the gradient is zero instead of inf.
        line.direction.sum().backward()
        expected = torch.zeros_like(p) if dtype == torch.float16 else torch.full_like(p, 1e12)
        self.assert_close(p1.grad, expected)

    def test_through_short_direction_is_unit(self, device, dtype):
        # A direction of norm 5 * 2**-20 (about 4.8e-6, exact float16 subnormals) is normalized as by
        # F.normalize(p=2, dim=-1): the 1e-12 norm floor stays below it.
        p0 = torch.zeros(2, device=device, dtype=dtype)
        p1 = torch.tensor([3.0 * 2**-20, -4.0 * 2**-20], device=device, dtype=dtype)
        line = ParametrizedLine.through(p0, p1)
        assert torch.equal(line.direction, torch.nn.functional.normalize(p1, p=2, dim=-1))
        self.assert_close(line.direction, torch.tensor([0.6, -0.8], device=device, dtype=dtype))

    def test_point_at(self, device, dtype):
        p0 = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        l1 = ParametrizedLine.through(p0, p1)
        self.assert_close(l1.point_at(0.0), torch.tensor([0.0, 0.0], device=device, dtype=dtype))
        self.assert_close(l1.point_at(0.5), torch.tensor([0.5, 0.0], device=device, dtype=dtype))
        self.assert_close(l1.point_at(1.0), torch.tensor([1.0, 0.0], device=device, dtype=dtype))

    def test_batched_point_at(self, device, dtype):
        origin = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        direction = torch.tensor([[1.0, 0.0], [1.0, 0.0]], device=device, dtype=dtype)
        steps = torch.tensor([2.0, 3.0], device=device, dtype=dtype)
        line = ParametrizedLine(origin, direction)
        expected = torch.stack([ParametrizedLine(origin[i], direction[i]).point_at(steps[i]) for i in range(2)])
        self.assert_close(line.point_at(steps), expected)
        # A (B, 1) step already carries the coordinate axis, and a Scalar is unwrapped to its tensor.
        self.assert_close(line.point_at(steps[:, None]), expected)
        from_scalar = line.point_at(Scalar(steps))
        assert type(from_scalar) is torch.Tensor
        self.assert_close(from_scalar, expected)

    def test_scalar_point_at_preserves_line_dtype(self, device, dtype):
        line = ParametrizedLine(
            torch.tensor([1.0, 2.0], device=device, dtype=dtype),
            torch.tensor([1.0, 0.0], device=device, dtype=dtype),
        )
        point = line.point_at(torch.tensor(2.0, device=device, dtype=torch.float32))
        self.assert_close(point, torch.tensor([3.0, 2.0], device=device, dtype=dtype))

    def test_projection1(self, device, dtype):
        p0 = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        p2 = torch.tensor([0.5, 0.5], device=device, dtype=dtype)
        p3_expected = torch.tensor([0.5, 0.0], device=device, dtype=dtype)
        l1 = ParametrizedLine.through(p0, p1)
        p3 = l1.projection(p2)
        self.assert_close(p3, p3_expected)

    def test_projection2(self, device, dtype):
        p0 = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        p2 = torch.tensor([0.5, 0.5], device=device, dtype=dtype)
        p3_expected = torch.tensor([0.0, 0.5], device=device, dtype=dtype)
        l1 = ParametrizedLine.through(p0, p1)
        p3 = l1.projection(p2)
        self.assert_close(p3, p3_expected)

    def test_projection(self, device, dtype):
        p0 = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        l1 = ParametrizedLine.through(p0, p1)
        point = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        point_projection = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        self.assert_close(l1.projection(point), point_projection)

    @pytest.mark.parametrize("batch_size", [2, 3])
    def test_batched_projection(self, device, dtype, batch_size):
        origin = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device, dtype=dtype)[:batch_size]
        direction = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], device=device, dtype=dtype)[:batch_size]
        point = torch.tensor([[2.0, 4.0], [6.0, 7.0], [8.0, 9.0]], device=device, dtype=dtype)[:batch_size]
        expected = torch.stack(
            [ParametrizedLine(origin[i], direction[i]).projection(point[i]) for i in range(batch_size)]
        )
        self.assert_close(ParametrizedLine(origin, direction).projection(point), expected)

    def test_distance(self, device, dtype):
        p0 = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        l1 = ParametrizedLine.through(p0, p1)
        point = torch.tensor([1.0, 4.0], device=device, dtype=dtype)
        distance_expected = torch.tensor(4.0, device=device, dtype=dtype)
        self.assert_close(l1.distance(point), distance_expected)

    def test_squared_distance(self, device, dtype):
        p0 = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        l1 = ParametrizedLine.through(p0, p1)
        point = torch.tensor([1.0, 4.0], device=device, dtype=dtype)
        distance_expected = torch.tensor(16.0, device=device, dtype=dtype)
        self.assert_close(l1.squared_distance(point), distance_expected)

    def test_distance_near_the_line_does_not_cancel_5016(self, device):
        # #5016: ||d||^2 - (d . u)^2 cancelled for points near a tilted line, so squared_distance went negative
        # and distance returned NaN, with a NaN gradient.
        o = torch.tensor([0.3, 0.7], device=device)
        line = ParametrizedLine(o, torch.tensor([0.28, 0.96], device=device))
        u = line.direction.detach()
        on_line = o + torch.linspace(-50.0, 50.0, 200, device=device)[:, None] * u
        # One value per row: a reduction over every element also passes the checks below.
        assert line.squared_distance(on_line).shape == (200,)
        assert line.distance(on_line).shape == (200,)
        assert (line.squared_distance(on_line) >= 0).all()
        assert torch.isfinite(line.distance(on_line)).all()
        assert line.distance(on_line).max() < 1e-4

        off_line = o + 40.0 * u + 1e-3 * torch.stack([-u[1], u[0]])
        self.assert_close(line.distance(off_line), torch.tensor(1e-3, device=device), rtol=1e-2, atol=0.0)

        p = on_line[7].clone().requires_grad_(True)
        line.distance(p).backward()
        assert torch.isfinite(p.grad).all()

    def test_through_coincident_points_raises_5041(self, device, dtype):
        # #5041: through(p, p) used to return a line with direction (0, 0).
        p = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        with pytest.raises(ValueCheckError, match="two distinct points"):
            ParametrizedLine.through(p, p.clone())

    def test_instersect_plane(self, device, dtype):
        p0 = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        l1 = ParametrizedLine.through(p0, p1)

        v0 = torch.tensor([3.0, 0.0, 1.0], device=device, dtype=dtype)
        v1 = torch.tensor([3.0, 1.0, 0.0], device=device, dtype=dtype)
        v2 = torch.tensor([3.0, 0.0, -1.0], device=device, dtype=dtype)
        pl0 = Hyperplane.through(v0, v1, v2)

        lmbda, point = l1.intersect(pl0)

        expected_point = torch.tensor([3.0, 0.0, 0.0], device=device, dtype=dtype)
        expected_lambda = torch.tensor(3.0, device=device, dtype=dtype)

        self.assert_close(lmbda, expected_lambda)
        self.assert_close(point, expected_point)

    def test_batched_intersect_plane(self, device, dtype):
        origin = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], device=device, dtype=dtype)
        direction = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device, dtype=dtype)
        normal = Vector3(torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype))
        plane = Hyperplane.from_vector(normal, Vector3(torch.tensor([3.0, 0.0, 0.0], device=device, dtype=dtype)))
        steps, points = ParametrizedLine(origin, direction).intersect(plane)
        expected_steps, expected_points = zip(
            *(ParametrizedLine(origin[i], direction[i]).intersect(plane) for i in range(2))
        )
        self.assert_close(steps, torch.stack(expected_steps))
        self.assert_close(points, torch.stack(expected_points))

    def test_intersect_plane_returns_tensors(self, device, dtype):
        plane = Hyperplane.through(
            torch.tensor([0.0, 0.0, 2.0], device=device, dtype=dtype),
            torch.tensor([1.0, 0.0, 2.0], device=device, dtype=dtype),
            torch.tensor([0.0, 1.0, 2.0], device=device, dtype=dtype),
        )
        line = ParametrizedLine(
            torch.tensor([1.0, 1.0, 0.0], device=device, dtype=dtype),
            torch.tensor([0.0, 0.6, 0.8], device=device, dtype=dtype),
        )

        lmbda, point = line.intersect(plane)

        assert type(lmbda) is torch.Tensor
        assert type(point) is torch.Tensor

        self.assert_close(lmbda, torch.tensor(2.5, device=device, dtype=dtype))
        self.assert_close(
            point,
            torch.tensor([1.0, 2.5, 2.0], device=device, dtype=dtype),
        )
        self.assert_close(plane.signed_distance(point).data, torch.zeros_like(lmbda))

        # An origin off z = 0, so that the sign of n . origin in lambda is pinned as well.
        line = ParametrizedLine(torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype), line.direction)
        lmbda, point = line.intersect(plane)
        self.assert_close(lmbda, torch.tensor(1.25, device=device, dtype=dtype))
        self.assert_close(point, torch.tensor([1.0, 1.75, 2.0], device=device, dtype=dtype))

    def test_intersect_plane_parallel(self, device, dtype):
        # the degenerate branch must return deterministic values, not uninitialized memory
        p0 = torch.tensor([0.0, 4.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([1.0, 4.0, 0.0], device=device, dtype=dtype)
        l1 = ParametrizedLine.through(p0, p1)

        v0 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        v1 = torch.tensor([1.0, 0.0, 1.0], device=device, dtype=dtype)
        v2 = torch.tensor([0.0, 1.0, 1.0], device=device, dtype=dtype)
        pl0 = Hyperplane.through(v0, v1, v2)

        lmbda, point = l1.intersect(pl0)

        self.assert_close(lmbda, torch.tensor(0.0, device=device, dtype=dtype))
        self.assert_close(point, p0)

    @pytest.mark.parametrize("parallel_dot", [0.0, 5e-7, -5e-7])
    @pytest.mark.parametrize("parallel_offset", [-1.0, 0.0])
    def test_intersect_plane_parallel_gradients(self, device, dtype, parallel_dot, parallel_offset):
        # The parallel fallback is locally (lambda, point) = (0, origin), so its Jacobian is zero except for
        # d point / d origin = I. An unselected division by zero used to poison all four input gradients.
        origin = torch.tensor([[0.0, 4.0, 0.0], [2.0, 3.0, 4.0]], device=device, dtype=dtype, requires_grad=True)
        direction = torch.tensor(
            [[1.0, 0.0, parallel_dot], [0.0, 0.0, -2.0]], device=device, dtype=dtype, requires_grad=True
        )
        normal = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype, requires_grad=True)
        offset = torch.tensor([parallel_offset, -2.0], device=device, dtype=dtype, requires_grad=True)
        plane = Hyperplane(Vector3(normal), Scalar(offset))

        lmbda, point = ParametrizedLine(origin, direction).intersect(plane)

        self.assert_close(lmbda, torch.tensor([0.0, 1.0], device=device, dtype=dtype))
        self.assert_close(point, torch.tensor([[0.0, 4.0, 0.0], [2.0, 3.0, 2.0]], device=device, dtype=dtype))
        (lmbda.sum() + point.sum()).backward()
        expected = (
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.5]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.5]],
            [[0.0, 0.0, 0.0], [-1.0, -1.5, -1.0]],
            [0.0, -0.5],
        )
        for tensor, gradient in zip((origin, direction, normal, offset), expected):
            self.assert_close(tensor.grad, torch.tensor(gradient, device=device, dtype=dtype))

    def test_intersect_plane_parallel_gradcheck(self, device):
        origin = torch.tensor([[0.0, 4.0, 0.0], [2.0, 3.0, 4.0]], device=device, dtype=torch.float64)
        direction = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -2.0]], device=device, dtype=torch.float64)
        normal = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device, dtype=torch.float64)
        offset = torch.tensor([-1.0, -2.0], device=device, dtype=torch.float64)

        def intersect(origin, direction, normal, offset):
            plane = Hyperplane(Vector3(normal), Scalar(offset))
            # Keep numerical perturbations inside the parallel branch, away from its threshold.
            return ParametrizedLine(origin, direction).intersect(plane, eps=1e-3)

        self.gradcheck(intersect, (origin, direction, normal, offset))

    def test_dynamo_intersect_plane_parallel_gradients(self, device, dtype, torch_optimizer):
        origin = torch.tensor([[0.0, 4.0, 0.0], [2.0, 3.0, 4.0]], device=device, dtype=dtype, requires_grad=True)
        direction = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -2.0]], device=device, dtype=dtype, requires_grad=True)
        normal = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype, requires_grad=True)
        offset = torch.tensor([-1.0, -2.0], device=device, dtype=dtype, requires_grad=True)
        inputs = (origin, direction, normal, offset)

        def intersect(origin, direction, normal, offset):
            return ParametrizedLine(origin, direction).intersect(Hyperplane(Vector3(normal), Scalar(offset)))

        expected_lambda, expected_point = intersect(*inputs)
        expected_gradients = torch.autograd.grad(expected_lambda.sum() + expected_point.sum(), inputs)
        actual_lambda, actual_point = torch_optimizer(intersect)(*inputs)
        actual_gradients = torch.autograd.grad(actual_lambda.sum() + actual_point.sum(), inputs)
        self.assert_close(actual_lambda, expected_lambda)
        self.assert_close(actual_point, expected_point)
        for actual, expected in zip(actual_gradients, expected_gradients):
            self.assert_close(actual, expected)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_gradcheck(self, device):
        pass

    def test_derived_state_moves_and_serializes(self, device, dtype):
        p0 = torch.rand(2, device=device, dtype=dtype, requires_grad=True)
        p1 = torch.rand(2, device=device, dtype=dtype, requires_grad=True)
        line = ParametrizedLine.through(p0, p1)
        assert line.direction.grad_fn is not None
        assert list(line.state_dict()) == ["_origin", "_direction"]
        origin = torch.zeros(2, device=device, dtype=dtype)
        restored = ParametrizedLine(origin, torch.ones(2, device=device, dtype=dtype))
        restored.load_state_dict(line.state_dict())
        self.assert_close(restored.direction, line.direction.detach())
        other = torch.float16 if dtype == torch.float32 else torch.float32  # float64 is unavailable on MPS
        moved = line.to(other)
        assert moved.origin.dtype == other
        assert moved.direction.dtype == other
        assert moved.direction.grad_fn is not None
        moved.direction.sum().backward()
        assert p1.grad is not None

    def test_user_leaf_receives_the_gradient(self, device, dtype):
        # A tensor that requires grad is kept, not re-wrapped as a new Parameter, so the gradient reaches it (#4943).
        origin = torch.tensor([0.5, 1.0], device=device, dtype=dtype, requires_grad=True)
        direction = torch.tensor([0.6, 0.8], device=device, dtype=dtype, requires_grad=True)
        line = ParametrizedLine(origin, direction)
        assert line.origin is origin
        assert line.direction is direction
        assert [name for name, _ in line.named_buffers()] == ["_origin", "_direction"]
        assert list(line.state_dict()) == ["_origin", "_direction"]
        line.point_at(2.0).sum().backward()
        assert origin.grad is not None
        assert direction.grad is not None
        self.assert_close(origin.grad, torch.ones_like(origin))
        self.assert_close(direction.grad, torch.full_like(direction, 2.0))
        # a tensor that does not require grad still becomes an optimizable parameter
        plain = ParametrizedLine(origin.detach().clone(), direction.detach().clone())
        assert [name for name, _ in plain.named_parameters()] == ["_origin", "_direction"]


def _near_vertical_points_5040(device, dtype) -> torch.Tensor:
    """Return 7 points (7, 2) off the line through (2, 0) along (0.1, 1), with perpendicular offsets (#5040)."""
    t = torch.tensor([-2.0, -1.3, -0.2, 0.4, 1.1, 2.7, 3.5], device=device, dtype=dtype)
    off = torch.tensor([0.21, -0.35, 0.12, 0.30, -0.27, 0.05, -0.18], device=device, dtype=dtype)
    u = torch.tensor([0.1, 1.0], device=device, dtype=dtype)
    u = u / u.norm()
    n = torch.stack([-u[1], u[0]])
    return torch.tensor([2.0, 0.0], device=device, dtype=dtype) + t[:, None] * u + off[:, None] * n


class TestFitLine(BaseTester):
    @pytest.mark.parametrize("B", [1, 2])
    @pytest.mark.parametrize("D", [2, 3, 4])
    def test_smoke(self, device, dtype, B, D):
        N: int = 10  # num points
        # A line needs distinct points: ones() is a set of identical points, rejected since #5041.
        t = torch.linspace(-1.0, 1.0, N, device=device, dtype=dtype)
        points = torch.stack([t] + [t * (i + 1) for i in range(D - 1)], dim=-1)[None].expand(B, N, D)
        line = fit_line(points)
        assert isinstance(line, ParametrizedLine)
        assert line.origin.shape == (B, D)
        assert line.direction.shape == (B, D)

        assert_close(line.origin, line[0])
        assert_close(line.direction, line[1])

        origin, direction = fit_line(points)
        assert_close(line.origin, origin)
        assert_close(line.direction, direction)

    def test_fit_line2(self, device, dtype):
        p0 = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([1.0, 1.0], device=device, dtype=dtype)

        l1 = ParametrizedLine.through(p0, p1)
        num_points: int = 10

        pts = []
        for t in torch.linspace(-10, 10, num_points):
            p2 = l1.point_at(t)
            pts.append(p2)
        pts = torch.stack(pts)

        line_est = fit_line(pts[None])
        dir_exp = torch.tensor([0.7071, 0.7071], device=device, dtype=dtype)
        # NOTE: for some reason the result in c[u/cuda differs
        angle_est = torch.nn.functional.cosine_similarity(line_est.direction, dir_exp, -1)

        angle_exp = torch.tensor([1.0], device=device, dtype=dtype)
        self.assert_close(angle_est.abs(), angle_exp)

    def test_fit_line3(self, device, dtype):
        p0 = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)

        l1 = ParametrizedLine.through(p0, p1)
        num_points: int = 10

        pts = []
        for t in torch.linspace(-10, 10, num_points):
            p2 = l1.point_at(t)
            pts.append(p2)
        pts = torch.stack(pts)

        line_est = fit_line(pts[None])
        dir_exp = torch.tensor([0.7071, 0.7071, 0.7071], device=device, dtype=dtype)
        # NOTE: result differs with the sign between cpu/cuda
        angle_est = torch.nn.functional.cosine_similarity(line_est.direction, dir_exp, -1)
        angle_exp = torch.tensor([1.0], device=device, dtype=dtype)
        self.assert_close(angle_est.abs(), angle_exp)

    def test_fit_line_weighted(self, device, dtype):
        p0 = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([1.0, 1.0], device=device, dtype=dtype)
        l1 = ParametrizedLine.through(p0, p1)

        num_points = 20
        ts = torch.linspace(-10, 10, num_points)
        points = torch.stack([l1.point_at(t) for t in ts])

        noise = torch.randn_like(points) * 0.05
        points_noisy = points + noise

        distances = torch.norm(points, dim=1)
        weights = torch.exp(-distances * 0.2)
        weights = weights / weights.max()

        line_est = fit_line(points_noisy[None], weights=weights[None])

        expected_dir = torch.tensor([0.7071, 0.7071], device=device, dtype=dtype)
        expected_dir = expected_dir / expected_dir.norm()

        angle_est = torch.nn.functional.cosine_similarity(line_est.direction, expected_dir, dim=-1)

        assert angle_est.abs() > 0.998

    def test_fit_line_weighted_3d_ignores_a_zero_weight_point_5014(self, device):
        # #5014: for D >= 3 the points were centred on the unweighted mean, so a point with weight 0 still moved
        # the origin and tilted the direction (by 16 degrees here). float32, not float64: MPS has no float64.
        d = torch.float32
        t = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], device=device, dtype=d)
        u = torch.tensor([2.0, 1.0, -2.0], device=device, dtype=d) / 3
        noise = torch.tensor(
            [
                [0.02, -0.01, 0.0],
                [-0.01, 0.02, 0.01],
                [0.0, 0.0, -0.02],
                [0.01, -0.02, 0.0],
                [-0.02, 0.01, 0.02],
                [0.0, 0.01, -0.01],
            ],
            device=device,
            dtype=d,
        )
        inliers = torch.tensor([1.0, 0.5, -1.0], device=device, dtype=d) + t[:, None] * u + noise
        outlier = torch.tensor([[6.0, -4.0, 5.0]], device=device, dtype=d)
        points = torch.cat([inliers, outlier])[None]
        weights = torch.tensor([[1.0] * 6 + [0.0]], device=device, dtype=d)

        expected = fit_line(inliers[None])
        actual = fit_line(points, weights)
        self.assert_close(actual.origin, expected.origin)
        # The direction is defined up to sign.
        self.assert_close((actual.direction * expected.direction).sum(-1).abs(), torch.ones(1, device=device, dtype=d))

        # Uniform weights give the unweighted fit.
        uniform = fit_line(points, torch.full((1, 7), 2.0, device=device, dtype=d))
        self.assert_close(uniform.origin, fit_line(points).origin)

        # Each batch row is centred on its own weighted centroid, whatever the other rows' weights sum to.
        batch = fit_line(torch.cat([points, points]), torch.cat([weights, 3.0 * weights]))
        self.assert_close(batch.origin, expected.origin.expand(2, 3))

        # A row whose weights are all 0 is rejected (#5041). With checks disabled, as under torch.compile, it keeps
        # the unweighted mean and does not break the other rows.
        zero_weights = torch.cat([weights, torch.zeros_like(weights)])
        with pytest.raises(ValueCheckError, match="positive sum of weights"):
            fit_line(torch.cat([points, points]), zero_weights)
        checks_were_enabled = are_checks_enabled()
        disable_checks()
        try:
            zero = fit_line(torch.cat([points, points]), zero_weights)
        finally:
            if checks_were_enabled:
                enable_checks()
        self.assert_close(zero.origin, torch.cat([expected.origin, points.mean(-2)]))

    def test_fit_line_vertical_dtype(self, device, dtype):
        pts = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]], device=device, dtype=dtype)
        line = fit_line(pts)
        assert line.origin.dtype == dtype
        assert line.direction.dtype == dtype
        self.assert_close(line.direction, torch.tensor([[0.0, 1.0]], device=device, dtype=dtype))

    def test_fit_line_2d_is_total_least_squares_5040(self, device, dtype):
        # #5040: the 2-D branch used to fit y-on-x ordinary least squares, several degrees off
        # a near-vertical set that the D >= 3 branch fits correctly.
        points = _near_vertical_points_5040(device, dtype)

        fit_2d = fit_line(points[None]).direction
        points_3d = torch.cat([points, torch.zeros(len(points), 1, device=device, dtype=dtype)], -1)
        fit_3d = fit_line(points_3d[None]).direction[..., :2]

        # same line up to the arbitrary SVD sign
        self.assert_close(fit_2d.abs(), fit_3d.abs())

        if dtype == torch.float64:
            # scaling the input must not change the fit (the old absolute 1e-8 vertical test
            # crossed at 1e-5); below float64 the scaled input itself is not representable
            scaled = fit_line((points * 1e-5)[None]).direction
            self.assert_close(scaled.abs(), fit_2d.abs())

    def test_fit_line_2d_symmetric_in_coordinates_5040(self, device, dtype):
        # #5040: swapping x and y must mirror the direction, not rotate it
        points = _near_vertical_points_5040(device, dtype)

        fit_2d = fit_line(points[None]).direction
        fit_swapped = fit_line(points.flip(-1)[None]).direction
        self.assert_close(fit_swapped, fit_2d.flip(-1))

    def test_fit_line_weighted_2d_total_least_squares_5040(self, device, dtype):
        # #5040: the weighted 2-D branch used to fit a weighted y-on-x slope; it must instead
        # use the weighted centroid and weighted second moments like the weighted D >= 3 branch.
        points = _near_vertical_points_5040(device, dtype)
        weights = torch.tensor([1.0, 2.0, 0.5, 1.5, 1.0, 2.0, 1.0], device=device, dtype=dtype)

        line = fit_line(points[None], weights[None])

        w = weights
        x_mean = (w * points[:, 0]).sum() / w.sum()
        y_mean = (w * points[:, 1]).sum() / w.sum()
        dx = points[:, 0] - x_mean
        dy = points[:, 1] - y_mean
        sxx = (w * dx * dx).sum()
        syy = (w * dy * dy).sum()
        sxy = (w * dx * dy).sum()
        theta = 0.5 * torch.atan2(2 * sxy, sxx - syy)
        expected_direction = torch.stack([theta.cos(), theta.sin()])[None]

        self.assert_close(line.direction, expected_direction)
        self.assert_close(line.origin, torch.stack([x_mean, y_mean])[None])

    def test_fit_line_2d_weight_scale_invariance(self, device, dtype):
        # Scaling all weights in a row changes neither its centroid nor the TLS direction.
        points = torch.tensor([[[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]]], device=device, dtype=dtype)
        weights = torch.tensor([[1.0, 2.0, 1.0]], device=device, dtype=dtype)
        # The largest weight is 2**(e - 1), so the weight sum 2**e overflows the dtype: normalising by the sum would not
        # help, normalising by the largest weight does.
        exponent = math.frexp(torch.finfo(dtype).max)[1] - 2
        scales = torch.tensor([2.0**exponent, 2.0**-exponent], device=device, dtype=dtype)[:, None]

        line = fit_line(points.expand(2, 3, 2), weights * scales)

        expected_origin = torch.tensor([[4.0, 8.0]], device=device, dtype=dtype).expand(2, 2)
        expected_direction = torch.tensor([[1.0 / math.sqrt(5), 2.0 / math.sqrt(5)]], device=device, dtype=dtype)
        self.assert_close(line.origin, expected_origin)
        self.assert_close(line.direction, expected_direction.expand(2, 2))
        assert line.origin.dtype == line.direction.dtype == dtype

    def test_fit_line_weighted_2d_float16_centroid(self, device):
        # Unit weights still overflow sum(w * offsets) for 256 float16 points in pixel coordinates.
        t = torch.arange(256, device=device, dtype=torch.float16)
        points = torch.stack([1000.0 + 4.0 * t, -500.0 + 2.0 * t], -1)[None]
        line = fit_line(points, torch.ones(1, 256, device=device, dtype=torch.float16))

        self.assert_close(line.origin, torch.tensor([[1510.0, -245.0]], device=device, dtype=torch.float16))
        direction = torch.tensor([[2.0 / math.sqrt(5), 1.0 / math.sqrt(5)]], device=device, dtype=torch.float16)
        self.assert_close(line.direction, direction)
        assert line.origin.dtype == line.direction.dtype == torch.float16

    def test_fit_line_weighted_2d_gradcheck(self, device):
        points = torch.tensor([[[0.0, 0.1], [1.0, 0.4], [2.0, 0.9], [3.0, 1.2]]], device=device)
        weights = torch.tensor([[1.0, 2.0, 1.0, 3.0]], device=device)

        def op(points, weights):
            return fit_line(points, weights).projection(points[:, 0])

        self.gradcheck(op, (points, weights), requires_grad=(True, True))

    @pytest.mark.parametrize("scale", [2.0**100, 2.0**-100], ids=["2**100", "2**-100"])
    def test_fit_line_weighted_2d_gradient_weight_scale_invariance(self, device, scale):
        # Scaling the weights changes neither the fit nor its gradient. Unnormalised float32 weights at these scales
        # gave a finite direction whose gradient with respect to the points was exactly zero.
        points = torch.tensor([[[0.0, 0.1], [1.0, 0.4], [2.0, 0.9], [3.0, 1.2]]], device=device)
        weights = torch.tensor([[1.0, 2.0, 1.0, 3.0]], device=device)

        def direction_grad(weights):
            points_ = points.clone().requires_grad_(True)
            fit_line(points_, weights).direction.sum().backward()
            return points_.grad

        expected = direction_grad(weights)
        assert expected.abs().amax() > 0.1
        self.assert_close(direction_grad(weights * scale), expected)

    @pytest.mark.parametrize("weighted", [False, True])
    def test_fit_line_2d_exactly_vertical_is_0_1_5040(self, device, dtype, weighted):
        # #5040: the direction of an exactly vertical line is (0, 1). Rounding in the mean used to leave a tiny
        # nonzero sxy of either sign, which gave (0, -1) for x = 0.7 or 7.7 in float64; and float32 atan2 returned
        # the float nearest to pi, whose half has a cosine of -4.4e-8 (vectorised CPU kernels for 8 rows or more,
        # and MPS). Eight rows, each vertical at its own x.
        x0 = torch.tensor([0.0, 0.1, 0.3, 0.7, -0.1, 7.7, 13.1, 100.3], device=device, dtype=dtype)
        y = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], device=device, dtype=dtype)
        points = torch.stack([x0[:, None].expand(8, 7), y.expand(8, 7)], -1)
        weights = torch.tensor([0.5, 2.0, 1.0, 1.5, 0.25, 1.0, 3.0], device=device, dtype=dtype).expand(8, 7)

        line = fit_line(points, weights if weighted else None)
        assert (line.direction[:, 0] >= 0).all(), line.direction
        self.assert_close(line.direction, torch.tensor([0.0, 1.0], device=device, dtype=dtype).expand(8, 2))
        self.assert_close(line.origin[:, 0], x0)

        # A row leaning left of vertical by far less than the rounding of pi / 2 gets (0, -1): its x component stays
        # non-negative, although float32 atan2 returns the float nearest to pi for it.
        leaning = points.clone()
        leaning[:, -1, 0] = leaning[:, -1, 0] - 1e-30 * (x0 == 0)
        line = fit_line(leaning, weights if weighted else None)
        assert (line.direction[:, 0] >= 0).all(), line.direction
        self.assert_close(line.direction.abs(), torch.tensor([0.0, 1.0], device=device, dtype=dtype).expand(8, 2))

    def test_fit_line_2d_vertical_compiled_half_5040(self, device, torch_optimizer):
        # Compiled half-precision kernels keep intermediates in float32 but round Python constants to the input
        # dtype: while theta was computed in the input dtype, an exactly vertical line got x = -4.8e-4 under
        # torch.compile.
        y = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        points = torch.stack([torch.full((7,), 0.7), y], -1)[None].to(device=device, dtype=torch.float16)
        direction = torch_optimizer(lambda p: fit_line(p).direction)(points)
        assert (direction[:, 0] >= 0).all(), direction
        self.assert_close(direction, torch.tensor([[0.0, 1.0]], device=device, dtype=torch.float16))

    def test_fit_line_2d_power_of_two_scale_5040(self, device, dtype):
        # #5040: the fit does not depend on the unit of the coordinates. Scaling by a power of two is exact in every
        # dtype; the scale is the square root of the dtype's range (256 in float16), where the second moments of the
        # scaled points used to overflow or underflow, and the direction came out vertical or NaN.
        points = _near_vertical_points_5040(device, dtype)[None]
        weights = torch.tensor([[1.0, 2.0, 0.5, 1.5, 1.0, 2.0, 1.0]], device=device, dtype=dtype)
        scale = 2.0 ** (math.frexp(torch.finfo(dtype).max)[1] // 2)
        scales = torch.tensor([scale, 1.0 / scale], device=device, dtype=dtype)[:, None, None]
        for w in (None, weights):
            line = fit_line(points, w)
            # Both scales in one batch: each row is rescaled on its own.
            scaled = fit_line(points * scales, None if w is None else w.expand(2, 7))
            self.assert_close(scaled.direction, line.direction.expand(2, 2))
            self.assert_close(scaled.origin / scales[..., 0], line.origin.expand(2, 2))

    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize(
        "dtype,scales", [(torch.float16, (256.0, 1.0 / 256.0)), (torch.float32, (2.0**64, 2.0**-64))]
    )
    def test_fit_line_3d_scale_invariance(self, device, weighted, dtype, scales):
        points = torch.tensor(
            [[[10.0, -5.0, 2.0], [11.0, -3.0, 5.0], [12.0, -0.5, 8.0], [13.0, 1.0, 11.0]]],
            device=device,
            dtype=dtype,
        )
        weights = torch.tensor([[1.0, 2.0, 1.0, 3.0]], device=device, dtype=points.dtype) if weighted else None
        expected = fit_line(points.float(), None if weights is None else weights.float())

        for scale in scales:
            actual = fit_line(points * scale, weights)
            self.assert_close(actual.direction.abs().float(), expected.direction.abs(), atol=1e-3, rtol=1e-3)
            self.assert_close(actual.origin.float() / scale, expected.origin, atol=1e-3, rtol=1e-3)

        if weighted and dtype == torch.float32:
            # 2**124 * 13 overflows float32 in sum(w p) unless the weights are normalised first.
            actual = fit_line(points, weights * 2.0**124)
            self.assert_close(actual.direction.abs(), expected.direction.abs())
            self.assert_close(actual.origin, expected.origin)

    def test_fit_line_3d_rows_rescaled_on_their_own(self, device):
        # Two rows of one batch, at coordinate scales 2**64 and 2**-64 and weight scales 2**100 and 2**-100: a scale
        # shared by the batch would underflow the second row's scatter matrix, or its weights to a zero sum.
        points = torch.tensor(
            [[10.0, -5.0, 2.0], [11.0, -3.0, 5.0], [12.0, -0.5, 8.0], [13.0, 1.0, 11.0]], device=device
        )
        weights = torch.tensor([1.0, 2.0, 1.0, 3.0], device=device)
        expected = fit_line(points[None], weights[None])
        s = torch.tensor([2.0**64, 2.0**-64], device=device)[:, None, None]
        ws = torch.tensor([2.0**100, 2.0**-100], device=device)[:, None]
        actual = fit_line(points * s, weights * ws)
        self.assert_close(actual.direction.abs(), expected.direction.abs().expand(2, 3))
        self.assert_close(actual.origin / s[..., 0], expected.origin.expand(2, 3))

    def test_fit_line_3d_weighted_float16_centroid_in_float32(self, device):
        # 128 float16 points near x = 1000: sum(w p) is 1.3e5, past the float16 maximum 65504, so the weighted
        # centroid is accumulated in float32. The output stays float16.
        # The float64 oracle is fitted on the CPU (MPS has no float64) from the float16-rounded points.
        t = torch.linspace(-1.0, 1.0, 128, dtype=torch.float64)
        points = torch.stack([1000.0 + 8.0 * t, 4.0 * t, -2.0 * t], -1)[None].half()
        expected = fit_line(points.double(), torch.ones(1, 128, dtype=torch.float64))
        actual = fit_line(points.to(device), torch.ones(1, 128, device=device, dtype=torch.float16))
        assert actual.origin.dtype == actual.direction.dtype == torch.float16
        self.assert_close(actual.direction.abs().cpu().double(), expected.direction.abs(), atol=1e-3, rtol=1e-3)
        self.assert_close(actual.origin.cpu().double(), expected.origin, atol=0.5, rtol=1e-3)

    def test_fit_line_3d_float16_centred_in_float32(self, device):
        # Unweighted float16 points near 1500, where the float16 spacing is 1: their centroid rounded to float16 is
        # about 0.45 off, which tilts the direction by about 6e-3. The centroid and the offsets are formed in float32.
        # The float64 oracle is fitted on the CPU (MPS has no float64) from the float16-rounded points.
        t = torch.linspace(-0.7, 1.0, 64, dtype=torch.float64)
        points = torch.stack([1500.0 + 6.0 * t, 1500.0 - 3.0 * t, 1500.0 + 2.0 * t], -1)[None].half()
        expected = fit_line(points.double())
        actual = fit_line(points.to(device))
        assert actual.direction.dtype == torch.float16
        self.assert_close(actual.direction.abs().cpu().double(), expected.direction.abs(), atol=1e-3, rtol=1e-3)

    def test_fit_line_3d_keeps_promoted_dtype(self, device):
        # Like the D = 2 branch, the line is returned in the promoted dtype of points and weights.
        points = torch.tensor(
            [[[0.0, 0.1, 0.3], [1.0, 0.4, 0.2], [2.0, 0.9, 0.1], [3.0, 1.6, 0.4], [4.0, 1.7, 0.3]]], device=device
        )
        f16, f32 = torch.float16, torch.float32
        for pdt, wdt in ((f16, f32), (f32, f16), (f16, f16)):
            weights = torch.ones(1, 5, device=device, dtype=wdt)
            line = fit_line(points.to(pdt), weights)
            expected = torch.promote_types(pdt, wdt)
            assert line.origin.dtype == line.direction.dtype == expected
            assert fit_line(points[..., :2].to(pdt), weights).origin.dtype == expected

    def test_fit_line_2d_degenerate_row_with_checks_disabled(self, device, dtype):
        # With checks disabled, as under torch.compile, identical 2-D points are not rejected: their scatter is 0,
        # and they get the direction (1, 0) rather than NaN, without touching the other rows.
        points = torch.tensor([[[0.0, 0.0], [1.0, 3.0], [2.0, 5.0]], [[1.0, 2.0]] * 3], device=device, dtype=dtype)
        weights = torch.tensor([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype)
        checks_were_enabled = are_checks_enabled()
        disable_checks()
        try:
            for w in (None, weights):
                line = fit_line(points, w)
                expected = fit_line(points[:1], None if w is None else w[:1])
                self.assert_close(line.direction[:1], expected.direction)
                self.assert_close(line.direction[1], torch.tensor([1.0, 0.0], device=device, dtype=dtype))
                self.assert_close(line.origin[1], points[1, 0])
        finally:
            if checks_were_enabled:
                enable_checks()

    def test_fit_line_3d_degenerate_row_with_checks_disabled(self, device, dtype):
        # With checks disabled, as under torch.compile, a row of identical 3-D points has a zero scatter matrix: it
        # is not divided by its zero scale, whose NaN would make the batched SVD raise for every row.
        points = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 3.0, 2.0], [2.0, 5.0, 3.0]], [[1.0, 2.0, 3.0]] * 3], device=device, dtype=dtype
        )
        weights = torch.tensor([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype)
        checks_were_enabled = are_checks_enabled()
        disable_checks()
        try:
            for w in (None, weights):
                line = fit_line(points, w)
                expected = fit_line(points[:1], None if w is None else w[:1])
                self.assert_close(line.direction[:1].abs(), expected.direction.abs())
                self.assert_close(line.origin[1], points[1, 0])
                assert torch.isfinite(line.direction[1]).all()
        finally:
            if checks_were_enabled:
                enable_checks()

    def test_fit_line_2d_steep_float16_5040(self, device, dtype):
        # #5040 (comment): a slope of 1e5 overflowed the float16 slope, giving NaN, or (1, 0) with weights.
        x = torch.linspace(0, 1e-3, 20, dtype=torch.float64)
        y = torch.linspace(0, 100, 20, dtype=torch.float64)
        points = torch.stack([x, y], -1)[None].to(device=device, dtype=dtype)
        expected = torch.tensor([[1e-5, 1.0]], device=device, dtype=dtype)
        for w in (None, torch.ones(1, 20, device=device, dtype=dtype)):
            self.assert_close(fit_line(points, w).direction, expected)

    def test_fit_line_degenerate_raises_5041(self, device, dtype):
        # #5041: a single point, identical points, or all-zero weights used to return an
        # arbitrary-looking line ((0, 1) / (1, 0, 0) directions, a NaN origin for zero weights).
        one_point = torch.tensor([[[1.0, 2.0]]], device=device, dtype=dtype)
        with pytest.raises(ValueCheckError, match="at least two points"):
            fit_line(one_point)

        identical = torch.tensor([[[1.0, 2.0, 3.0]] * 5], device=device, dtype=dtype)
        with pytest.raises(ValueCheckError, match="two distinct points"):
            fit_line(identical)

        # Identical points whose mean rounds: three copies of 0.1 sum to 0.30000000000000004, so centring on the
        # mean leaves a residual of about 1e-17 and a mean-based test would accept them.
        identical_rounding = torch.tensor([[[0.1, 0.7]] * 3], device=device, dtype=dtype)
        with pytest.raises(ValueCheckError, match="two distinct points"):
            fit_line(identical_rounding)

        # A batch is rejected as a whole when one of its rows is degenerate.
        batch = torch.tensor([[[0.0, 0.0], [1.0, 3.0]], [[1.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        with pytest.raises(ValueCheckError, match="two distinct points"):
            fit_line(batch)

        # #5041: all-zero weights used to return an arbitrary line — a NaN origin and (1, 0) in 2-D,
        # the unweighted mean and the first singular vector of a zero matrix in 3-D. The row is rejected
        # like any other degenerate one, so a batch with one all-zero-weights row is rejected as a whole.
        points_3d = torch.tensor([[[0.0, 0.0, 0.3], [1.0, 3.0, -0.2], [2.0, 5.0, 0.1]]], device=device, dtype=dtype)
        zero_weights = torch.zeros(1, 3, device=device, dtype=dtype)
        mixed = torch.cat([torch.ones_like(zero_weights), zero_weights])
        for points in (points_3d[..., :2], points_3d):
            with pytest.raises(ValueCheckError, match="positive sum of weights"):
                fit_line(points, zero_weights)
            with pytest.raises(ValueCheckError, match="positive sum of weights"):
                fit_line(torch.cat([points, points]), mixed)

        # Weights that are not a tensor still fail the type check, not the weight-sum check.
        with pytest.raises(TypeCheckError, match="weights must be a tensor"):
            fit_line(points_3d, [[1.0, 1.0, 1.0]])

    def test_fit_line_weighted_identical_points_raise_5082(self, device, dtype):
        # #5082: a point with weight 0 does not count, so weights that are positive on a single point, or only on
        # copies of one point, leave nothing to fit. Such a row used to return the fallback direction, (0, 1) in 2-D
        # and (1, 0, 0) in 3-D, whatever the points were.
        points_3d = torch.tensor([[[0.0, 0.0, 0.3], [1.0, 0.4, -0.2], [2.5, 0.9, 0.1]]], device=device, dtype=dtype)
        copies_3d = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 5.0, -1.0], [1.0, 2.0, 3.0]]], device=device, dtype=dtype)
        single = ([1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [-1.0, 0.0, 2.0])
        for points in (points_3d[..., :2], points_3d):
            for w in single:
                weights = torch.tensor([w], device=device, dtype=dtype)
                with pytest.raises(ValueCheckError, match="two distinct points with positive weight"):
                    fit_line(points, weights)
                # a batch is rejected as a whole when one of its rows is degenerate
                with pytest.raises(ValueCheckError, match="two distinct points with positive weight"):
                    fit_line(torch.cat([points, points]), torch.cat([torch.ones_like(weights), weights]))
            # two distinct points with positive weight determine the line through them
            line = fit_line(points, torch.tensor([[1.0, 0.0, 1.0]], device=device, dtype=dtype))
            expected = fit_line(points[:, [0, 2]])
            self.assert_close(line.origin, expected.origin)
            self.assert_close(
                (line.direction * expected.direction).sum(-1).abs(), torch.ones(1, device=device, dtype=dtype)
            )

        for points in (copies_3d[..., :2], copies_3d):
            with pytest.raises(ValueCheckError, match="two distinct points with positive weight"):
                fit_line(points, torch.tensor([[1.0, 0.0, 3.0]], device=device, dtype=dtype))
            fit_line(points, torch.tensor([[1.0, 0.5, 3.0]], device=device, dtype=dtype))

        # Two points with positive weight that differ in one coordinate only are distinct: they give the vertical
        # line in 2-D and the line along z in 3-D.
        axis_3d = torch.tensor([[[1.0, 2.0, 3.0], [9.0, 9.0, 9.0], [1.0, 2.0, 4.0]]], device=device, dtype=dtype)
        for points, direction in ((axis_3d[..., [0, 2]], [0.0, 1.0]), (axis_3d, [0.0, 0.0, 1.0])):
            line = fit_line(points, torch.tensor([[1.0, 0.0, 1.0]], device=device, dtype=dtype))
            self.assert_close(line.direction.abs(), torch.tensor([direction], device=device, dtype=dtype))

    def test_fit_line_negative_weights_raise_5106(self, device, dtype):
        # #5106: a negative weight passes the weight-sum check, but it can make the weighted scatter indefinite. The
        # D >= 3 branch then takes the axis of the largest |eigenvalue| and a 2-D total least squares fit the axis of
        # the largest eigenvalue: 90 degrees apart for weights (1, -3, 1, 2) on these points. Any negative weight is
        # rejected, also one such as -0.01 that leaves the scatter without a negative eigenvalue.
        points_2d = torch.tensor([[[0.0, 0.0], [1.0, 0.4], [2.5, 0.9], [3.0, 2.0]]], device=device, dtype=dtype)
        points_3d = torch.nn.functional.pad(points_2d, (0, 1))
        ones = torch.ones(1, 4, device=device, dtype=dtype)
        for points in (points_2d, points_3d):
            for w in ([1.0, -3.0, 1.0, 2.0], [1.0, -0.01, 1.0, 1.0]):
                weights = torch.tensor([w], device=device, dtype=dtype)
                with pytest.raises(ValueCheckError, match="non-negative weights"):
                    fit_line(points, weights)
                # a batch is rejected as a whole when one of its rows has a negative weight
                with pytest.raises(ValueCheckError, match="non-negative weights"):
                    fit_line(torch.cat([points, points]), torch.cat([ones, weights]))

            # A zero weight, +0.0 or -0.0, is not negative: it drops its point.
            expected = fit_line(points[:, [0, 2, 3]])
            for zero in (0.0, -0.0):
                line = fit_line(points, torch.tensor([[1.0, zero, 1.0, 1.0]], device=device, dtype=dtype))
                self.assert_close(line.origin, expected.origin)
                self.assert_close(
                    (line.direction * expected.direction).sum(-1).abs(), torch.ones(1, device=device, dtype=dtype)
                )

    def test_dynamo_skips_degenerate_checks(self, device, dtype, torch_optimizer):
        # The degeneracy checks depend on tensor values, so they are skipped under torch.compile: a compiled call
        # on identical points returns what an eager call returns with checks disabled.
        p = torch.tensor([[[1.0, 2.0, 3.0]] * 4], device=device, dtype=dtype)

        def op(points):
            through = ParametrizedLine.through(points[0, 0], points[0, 1]).direction
            return through, fit_line(points).direction, fit_line(points[..., :2]).direction

        actual = torch_optimizer(op)(p)
        checks_were_enabled = are_checks_enabled()
        disable_checks()
        try:
            expected = op(p)
        finally:
            if checks_were_enabled:
                enable_checks()
        self.assert_close(actual[0], expected[0])
        self.assert_close(actual[1], expected[1])
        self.assert_close(actual[2], expected[2])

    def test_fit_line_small_valid_set_still_fits(self, device, dtype):
        # The degeneracy test is relative, not absolute: a small-but-distinct set still fits.
        points = torch.tensor([[[1e-6, 2e-6], [3e-6, 7e-6]]], device=device, dtype=dtype)
        line = fit_line(points)
        assert line.direction.shape == (1, 2)
        assert torch.isfinite(line.direction).all()
        assert torch.isfinite(line.origin).all()

    @pytest.mark.parametrize("dim", [2, 3])
    def test_gradcheck(self, device, dim):
        # Two point sets whose rows differ, each projected onto its own fitted line (#5013).
        def proxy_func(pts, weights):
            return fit_line(pts, weights).projection(pts[:, 0])

        pts = torch.rand(2, 5, dim, device=device)
        weights = torch.rand(2, 5, device=device)
        self.gradcheck(proxy_func, (pts, weights), requires_grad=(True, True))

    @pytest.mark.parametrize("dim", [3, 4])
    def test_weighted_fit_saves_linear_storage(self, device, dtype, dim):
        # A differentiable fit needs storage proportional to the points, not an N-by-N weight matrix.
        points = torch.rand(2, 128, dim, device=device, dtype=dtype, requires_grad=True)
        weights = torch.rand(2, 128, device=device, dtype=dtype, requires_grad=True)
        saved_sizes = []

        def pack(tensor):
            saved_sizes.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            line = fit_line(points, weights)
            loss = line.projection(points[:, 0]).square().sum()
            loss.backward()

        assert max(saved_sizes) <= 8 * points.numel()
        assert torch.isfinite(points.grad).all()
        assert torch.isfinite(weights.grad).all()

    def test_dynamo_weighted_fit_3d(self, device, dtype, torch_optimizer):
        points = torch.rand(2, 32, 3, device=device, dtype=dtype)
        weights = torch.rand(2, 32, device=device, dtype=dtype)

        def op(points, weights):
            line = fit_line(points, weights)
            return line.projection(points[:, 0])

        self.assert_close(torch_optimizer(op)(points, weights), op(points, weights))

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestConventionsParametrizedLine(BaseTester):
    def test_convention_parametrized_line_direction_is_not_normalized(self, device, dtype):
        # The constructor does not normalise the direction, so point_at(t) = origin + t * direction steps in units of
        # ||direction||. through(p0, p1) normalises p1 - p0, so there t is the Euclidean distance from p0. The
        # distance methods assume a unit direction, which the constructor does not enforce.
        origin = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        direction = torch.tensor([3.0, 4.0], device=device, dtype=dtype)
        assert torch.linalg.vector_norm(direction).item() == 5.0  # not a unit direction

        line = ParametrizedLine(origin, direction)
        self.assert_close(line.direction, direction)
        self.assert_close(line.point_at(1.0), torch.tensor([4.0, 6.0], device=device, dtype=dtype))

        through = ParametrizedLine.through(origin, origin + direction)
        self.assert_close(through.origin, origin)
        self.assert_close(through.direction, torch.tensor([0.6, 0.8], device=device, dtype=dtype))
        self.assert_close(through.point_at(5.0), torch.tensor([4.0, 6.0], device=device, dtype=dtype))

    def test_convention_parametrized_line_intersect_lambda_units(self, device, dtype):
        # intersect returns (lambda, point) with point = point_at(lambda), so lambda is in units of the stored
        # direction: a unit direction gives the Euclidean distance from the origin, a non-unit direction a different
        # lambda for the same point, and the reversed direction a negative one. A non-unit normal gives the same
        # lambda. lambda = -(offset + n . origin) / (n . direction).
        normal = torch.tensor([1.0, 2.0, 2.0], device=device, dtype=dtype) / 3
        assert (normal.abs() >= 0.1).all()  # a tilted plane: no normal component near 0
        plane = Hyperplane.from_vector(Vector3(normal), Vector3(torch.ones(3, device=device, dtype=dtype)))
        origin = torch.tensor([0.5, -1.0, 2.0], device=device, dtype=dtype)
        direction = torch.tensor([2.0, 1.0, -0.5], device=device, dtype=dtype)
        # origin + 5/6 * direction lies on the plane; ||direction|| = sqrt(5.25).
        expected_point = torch.tensor([13 / 6, -1 / 6, 19 / 12], device=device, dtype=dtype)

        lmbda, point = ParametrizedLine(origin, direction / torch.linalg.vector_norm(direction)).intersect(plane)
        self.assert_close(lmbda, torch.tensor(5.25**0.5 * 5 / 6, device=device, dtype=dtype))  # 1.9094
        self.assert_close(point, expected_point)
        self.assert_close(plane.signed_distance(point).data, torch.zeros((), device=device, dtype=dtype))

        lmbda, point = ParametrizedLine(origin, direction).intersect(plane)
        self.assert_close(lmbda, torch.tensor(5 / 6, device=device, dtype=dtype))  # 0.8333
        self.assert_close(point, expected_point)

        lmbda, point = ParametrizedLine(origin, -direction).intersect(plane)
        self.assert_close(lmbda, torch.tensor(-5 / 6, device=device, dtype=dtype))
        self.assert_close(point, expected_point)

        nonunit_plane = Hyperplane.from_vector(Vector3(3 * normal), Vector3(torch.ones(3, device=device, dtype=dtype)))
        lmbda, point = ParametrizedLine(origin, direction).intersect(nonunit_plane)
        self.assert_close(lmbda, torch.tensor(5 / 6, device=device, dtype=dtype))
        self.assert_close(point, expected_point)


class TestConventionsFitLine(BaseTester):
    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_convention_fit_line_centroid_and_unit_direction(self, device, dtype, dim, weighted):
        # fit_line returns, for each batch row on its own, a line through the centroid of that row's points (the
        # weighted centroid sum(w p) / sum(w) with weights) and a unit direction, for D = 2 and D >= 3 alike. The
        # weights are uneven, so the two centroids differ, and the points are off any single line. Each row's direction
        # is compared with the fit of that row alone, up to sign (the D >= 3 sign is unspecified).
        points = torch.tensor(
            [[0.0, 0.0, 0.3], [1.0, 0.4, -0.2], [2.5, 0.9, 0.1], [3.0, 1.6, 0.4], [4.2, 1.7, -0.3]],
            device=device,
            dtype=dtype,
        )[:, :dim]
        rows = torch.stack([points, 2.0 * points.flip(-1) + 1.0])  # the second row: another line elsewhere
        w = torch.tensor([[3.0, 0.5, 1.0, 0.25, 2.0], [0.5, 2.0, 1.0, 4.0, 0.25]], device=device, dtype=dtype)

        line = fit_line(rows, w if weighted else None)
        for i in range(2):
            centroid = (w[i, :, None] * rows[i]).sum(0) / w[i].sum() if weighted else rows[i].mean(0)
            self.assert_close(line.origin[i], centroid)
            one = torch.ones((), device=device, dtype=dtype)
            self.assert_close(torch.linalg.vector_norm(line.direction[i]), one)
            single = fit_line(rows[i : i + 1], w[i : i + 1] if weighted else None).direction[0]
            self.assert_close(line.direction[i], torch.sign((line.direction[i] * single).sum()) * single)
