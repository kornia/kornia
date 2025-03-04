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
from torch.autograd import gradcheck

from kornia.geometry.line import ParametrizedLine, fit_line
from kornia.geometry.plane import Hyperplane

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

    def test_point_at(self, device, dtype):
        p0 = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        l1 = ParametrizedLine.through(p0, p1)
        self.assert_close(l1.point_at(0.0), torch.tensor([0.0, 0.0], device=device, dtype=dtype))
        self.assert_close(l1.point_at(0.5), torch.tensor([0.5, 0.0], device=device, dtype=dtype))
        self.assert_close(l1.point_at(1.0), torch.tensor([1.0, 0.0], device=device, dtype=dtype))

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


class TestFitLine(BaseTester):
    @pytest.mark.parametrize("B", (1, 2))
    @pytest.mark.parametrize("D", (2, 3, 4))
    def test_smoke(self, device, dtype, B, D):
        N: int = 10  # num points
        points = torch.ones(B, N, D, device=device, dtype=dtype)
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

    @pytest.mark.skip(reason="numerical do not match with analytical")
    def test_gradcheck(self, device):
        def proxy_func(pts, weights):
            line = fit_line(pts, weights)
            return line.projection(pts[:, 0].T)

        pts = torch.rand(1, 3, 2, device=device, dtype=torch.float64, requires_grad=True)
        weights = torch.rand(1, 3, device=device, dtype=torch.float64, requires_grad=False)
        assert gradcheck(proxy_func, (pts, weights), raise_exception=True)

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
