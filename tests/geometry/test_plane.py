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
from kornia.core.exceptions import BaseError, ValueCheckError
from kornia.geometry.plane import Hyperplane, fit_plane
from kornia.geometry.vector import Vector3

from testing.base import BaseTester


# TODO: implement the rest of methods
class TestFitPlane(BaseTester):
    @pytest.mark.parametrize("N", (4, 10))
    @pytest.mark.parametrize("D", (3,))
    # @pytest.mark.parametrize("D", (2, 3, 4))
    def test_smoke(self, device, dtype, N, D):
        # A plane needs non-collinear points: ones() is a set of identical points, rejected since #5041.
        t = torch.linspace(-1.0, 1.0, N, device=device, dtype=dtype)
        points = torch.stack([t, t**2, 1.0 - t], dim=-1).expand(N, D).contiguous()
        plane = fit_plane(points)
        assert isinstance(plane, Hyperplane)
        assert plane.offset.shape == ()
        assert plane.normal.shape == (D,)

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

    def test_fit_plane_degenerate_raises_5041(self, device, dtype):
        # #5041: fewer than three points, collinear points or identical points used to
        # return an arbitrary valid-looking plane normal.
        a = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
        b = torch.tensor([2.0, 3.0, 4.0], device=device, dtype=dtype)
        c = 2 * b - a
        with pytest.raises(ValueCheckError, match="at least three points"):
            fit_plane(a[None])
        with pytest.raises(ValueCheckError, match="at least three points"):
            fit_plane(torch.stack([a, b]))
        with pytest.raises(ValueCheckError, match="not collinear"):
            fit_plane(torch.stack([a, b, c]))
        with pytest.raises(ValueCheckError, match="not identical"):
            fit_plane(torch.stack([a, a.clone(), a.clone()]))
        # Collinear along an axis: the second singular value is exactly 0.
        on_axis = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device=device, dtype=dtype)
        with pytest.raises(ValueCheckError, match="not collinear"):
            fit_plane(on_axis)
        # Identical points whose mean rounds are still rejected as identical.
        rounding = torch.tensor([0.1, 0.7, 0.3], device=device, dtype=dtype)
        with pytest.raises(ValueCheckError, match="not identical"):
            fit_plane(rounding.expand(3, 3))

    def test_fit_plane_small_valid_set_still_fits(self, device, dtype):
        # The collinearity test is relative, not absolute: a small non-degenerate set still fits.
        s = 1e-6
        points = torch.tensor([[0.0, 0.0, 0.0], [s, 0.0, 0.0], [0.0, s, 0.0]], device=device, dtype=dtype)
        plane = fit_plane(points)
        assert plane.normal.shape == (3,)
        assert torch.isfinite(plane.normal.unwrap()).all()


# TODO: implement the rest of methods
class TestHyperplane(BaseTester):
    @pytest.mark.parametrize("shape", (None, (1,), (2, 1)))
    def test_smoke(self, device, dtype, shape):
        p0 = Vector3.random(shape, device, dtype)
        n0 = Vector3.random(shape, device, dtype).normalized()
        pl0 = Hyperplane.from_vector(n0, p0)
        assert pl0.normal.shape == shape or (3,)
        assert pl0.offset.shape == ((*shape,) if shape is not None else ())

    def test_serialization(self, device, dtype, tmp_path):
        p = Vector3.random((), device, dtype)
        n = Vector3.random((), device, dtype).normalized()
        plane = Hyperplane.from_vector(n, p)

        file_path = tmp_path / "plane.pt"
        torch.save(plane, file_path)
        assert file_path.is_file()

        loaded_plane = torch.load(file_path, weights_only=False)
        self.assert_close(plane.normal.unwrap(), loaded_plane.normal.unwrap())

    @pytest.mark.parametrize("shape", ((2,), (1, 2), (3,), (1, 3)))
    def test_through_two_points_raises(self, device, dtype, shape):
        # Hyperplane stores a Vector3 normal and has no 2D form: two points must be rejected with a
        # message that names the three-point requirement, whatever the points' dimension.
        p0 = torch.rand(shape, device=device, dtype=dtype)
        p1 = torch.rand(shape, device=device, dtype=dtype)
        with pytest.raises(BaseError, match="requires three points"):
            Hyperplane.through(p0, p1)

    @pytest.mark.parametrize("shape", (None, (1,), (2, 1)))
    def test_through_three(self, device, dtype, shape):
        v0 = Vector3.random(shape, device, dtype)
        v1 = Vector3.random(shape, device, dtype)
        v2 = Vector3.random(shape, device, dtype)
        # TODO: improve api so that we can accept Vector too
        p0 = Hyperplane.through(v0, v1, v2)
        assert p0.normal.shape == shape or (3,)
        assert p0.offset.shape == ((*shape,) if shape is not None else ())

    def test_through_collinear_points_raises_5041(self, device, dtype):
        # #5041: collinear (or coincident) points used to take the SVD fallback and return an
        # arbitrary plane containing the line instead of raising.
        a = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
        b = torch.tensor([2.0, 3.0, 4.0], device=device, dtype=dtype)
        c = 2 * b - a
        with pytest.raises(ValueCheckError, match="not collinear"):
            Hyperplane.through(a, b, c)
        with pytest.raises(ValueCheckError, match="not collinear"):
            Hyperplane.through(a, a.clone(), a.clone())
        # A batch is rejected as a whole when one of its rows is degenerate.
        p0 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([0.0, 2.0, 0.0], device=device, dtype=dtype)
        p2 = torch.tensor([0.0, 0.0, 3.0], device=device, dtype=dtype)
        with pytest.raises(ValueCheckError, match="not collinear"):
            Hyperplane.through(torch.stack([p0, a]), torch.stack([p1, b]), torch.stack([p2, c]))

    @pytest.mark.parametrize("scale", (1.0, 1e-4))
    def test_through_small_valid_triangle_still_fits(self, device, dtype, scale):
        # A small triangle is not collinear. In float16 the cross product of the 1e-4 triangle underflows to 0
        # (1e-8 is below the smallest subnormal), so it reaches the SVD fallback, which must not reject it.
        p0 = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([scale, 0.0, 0.0], device=device, dtype=dtype)
        p2 = torch.tensor([0.0, scale, 0.0], device=device, dtype=dtype)
        plane = Hyperplane.through(p0, p1, p2)
        self.assert_close(plane.normal.unwrap().abs(), torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype))

    def test_dynamo_skips_degenerate_checks(self, device, dtype, torch_optimizer):
        # The degeneracy checks depend on tensor values, so they are skipped under torch.compile: a compiled call
        # on collinear points returns what an eager call returns with checks disabled.
        a = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
        b = torch.tensor([2.0, 3.0, 4.0], device=device, dtype=dtype)
        c = 2 * b - a

        def op(p0, p1, p2):
            return Hyperplane.through(p0, p1, p2).normal.unwrap(), fit_plane(torch.stack([p0, p1, p2])).normal.unwrap()

        actual = torch_optimizer(op)(a, b, c)
        checks_were_enabled = are_checks_enabled()
        disable_checks()
        try:
            expected = op(a, b, c)
        finally:
            if checks_were_enabled:
                enable_checks()
        self.assert_close(actual[0], expected[0])
        self.assert_close(actual[1], expected[1])

    def test_thin_valid_triangle_still_fits(self, device, dtype):
        # The collinearity tolerance scales with the dtype: a sliver whose height is 64 machine epsilons of its
        # base is a valid plane in every dtype (in float64 that is 1.4e-14, far below a fixed 1e-6 tolerance).
        h = 64 * torch.finfo(dtype).eps
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        self.assert_close(Hyperplane.through(points[0], points[1], points[2]).normal.unwrap().abs(), expected)
        self.assert_close(fit_plane(points).normal.unwrap().abs(), expected)

    @pytest.mark.parametrize("shape", (None, (1,), (2, 1)))
    def test_abs_signed_distance(self, device, dtype, shape):
        p0 = Vector3.random(shape, device, dtype)
        p1 = Vector3.random(shape, device, dtype)

        n0 = Vector3.random(shape, device, dtype).normalized()
        n1 = Vector3.random(shape, device, dtype).normalized()

        s0 = torch.rand(shape or (), device=device, dtype=dtype)
        s1 = torch.rand(shape or (), device=device, dtype=dtype)

        pl0 = Hyperplane.from_vector(n0, p0)
        pl1 = Hyperplane.from_vector(n1, p1)

        expected = torch.ones(shape or (), device=device, dtype=dtype)
        self.assert_close(pl1.signed_distance(p1 + n1 * s0[..., None]), s0)
        assert (pl0.abs_distance(p0) < expected).all()
        assert (pl1.signed_distance(pl1.projection(p0)) < expected).all()
        assert (pl1.abs_distance(p1 + pl1.normal * s1) < expected).all()

    def test_projection(self, device, dtype):
        v0 = Vector3.from_coords(0.0, 0.0, 0.0, device=device, dtype=dtype)
        v1 = Vector3.from_coords(0.0, 1.0, 0.0, device=device, dtype=dtype)
        v2 = Vector3.from_coords(0.0, 0.0, 1.0, device=device, dtype=dtype)
        plane_in_world = Hyperplane.through(v0, v1, v2)
        p_in_world = Vector3.from_coords(0.0, 0.0, 1.0, device=device, dtype=dtype)
        p_in_plane = plane_in_world.projection(p_in_world)
        p_in_plane_expected = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        self.assert_close(p_in_plane, p_in_plane_expected)

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
