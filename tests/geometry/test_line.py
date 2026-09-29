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

    @pytest.mark.parametrize("batch_size", (2, 3))
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
        assert moved.origin.dtype == other and moved.direction.dtype == other
        assert moved.direction.grad_fn is not None
        moved.direction.sum().backward()
        assert p1.grad is not None

    def test_user_leaf_receives_the_gradient(self, device, dtype):
        # A tensor that requires grad is kept, not re-wrapped as a new Parameter, so the gradient reaches it (#4943).
        origin = torch.tensor([0.5, 1.0], device=device, dtype=dtype, requires_grad=True)
        direction = torch.tensor([0.6, 0.8], device=device, dtype=dtype, requires_grad=True)
        line = ParametrizedLine(origin, direction)
        assert line.origin is origin and line.direction is direction
        assert [name for name, _ in line.named_buffers()] == ["_origin", "_direction"]
        assert list(line.state_dict()) == ["_origin", "_direction"]
        line.point_at(2.0).sum().backward()
        assert origin.grad is not None and direction.grad is not None
        self.assert_close(origin.grad, torch.ones_like(origin))
        self.assert_close(direction.grad, torch.full_like(direction, 2.0))
        # a tensor that does not require grad still becomes an optimizable parameter
        plain = ParametrizedLine(origin.detach().clone(), direction.detach().clone())
        assert [name for name, _ in plain.named_parameters()] == ["_origin", "_direction"]


class TestFitLine(BaseTester):
    @pytest.mark.parametrize("B", (1, 2))
    @pytest.mark.parametrize("D", (2, 3, 4))
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

    def test_dynamo_skips_degenerate_checks(self, device, dtype, torch_optimizer):
        # The degeneracy checks depend on tensor values, so they are skipped under torch.compile: a compiled call
        # on identical points returns what an eager call returns with checks disabled.
        p = torch.tensor([[[1.0, 2.0, 3.0]] * 4], device=device, dtype=dtype)

        def op(points):
            return ParametrizedLine.through(points[0, 0], points[0, 1]).direction, fit_line(points).direction

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

    def test_fit_line_small_valid_set_still_fits(self, device, dtype):
        # The degeneracy test is relative, not absolute: a small-but-distinct set still fits.
        points = torch.tensor([[[1e-6, 2e-6], [3e-6, 7e-6]]], device=device, dtype=dtype)
        line = fit_line(points)
        assert line.direction.shape == (1, 2)
        assert torch.isfinite(line.direction).all()
        assert torch.isfinite(line.origin).all()

    @pytest.mark.parametrize("dim", (2, 3))
    def test_gradcheck(self, device, dim):
        # Two point sets whose rows differ, each projected onto its own fitted line (#5013).
        def proxy_func(pts, weights):
            return fit_line(pts, weights).projection(pts[:, 0])

        pts = torch.rand(2, 5, dim, device=device)
        weights = torch.rand(2, 5, device=device)
        self.gradcheck(proxy_func, (pts, weights), requires_grad=(True, False))

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
