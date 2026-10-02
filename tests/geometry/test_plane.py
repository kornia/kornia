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
    @pytest.mark.parametrize("N", [4, 10])
    @pytest.mark.parametrize("D", [3])
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
    @pytest.mark.parametrize("shape", [None, (1,), (2, 1)])
    def test_smoke(self, device, dtype, shape):
        p0 = Vector3.random(shape, device, dtype)
        n0 = Vector3.random(shape, device, dtype).normalized()
        pl0 = Hyperplane.from_vector(n0, p0)
        assert pl0.normal.shape == ((*shape, 3) if shape is not None else (3,))
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

    @pytest.mark.parametrize("shape", [(2,), (1, 2), (3,), (1, 3)])
    def test_through_two_points_raises(self, device, dtype, shape):
        # Hyperplane stores a Vector3 normal and has no 2D form: two points must be rejected with a
        # message that names the three-point requirement, whatever the points' dimension.
        p0 = torch.rand(shape, device=device, dtype=dtype)
        p1 = torch.rand(shape, device=device, dtype=dtype)
        with pytest.raises(BaseError, match="requires three points"):
            Hyperplane.through(p0, p1)

    @pytest.mark.parametrize("shape", [None, (1,), (2, 1)])
    def test_through_three(self, device, dtype, shape):
        v0 = Vector3.random(shape, device, dtype)
        v1 = Vector3.random(shape, device, dtype)
        v2 = Vector3.random(shape, device, dtype)
        # TODO: improve api so that we can accept Vector too
        p0 = Hyperplane.through(v0, v1, v2)
        assert p0.normal.shape == ((*shape, 3) if shape is not None else (3,))
        assert p0.offset.shape == ((*shape,) if shape is not None else ())

    def test_through_orthogonal_equal_length_gradient_5056(self, device, dtype):
        # #5056: the SVD fallback ran on every row and torch.where only zeroed its gradient. When p2 - p0 and
        # p1 - p0 are orthogonal and of equal length the two singular values coincide, the SVD backward divides by
        # their difference, and 0 * inf turned every input gradient into nan although the plane came from the cross
        # product. Row 0: v0 = (2, 1, -2) and v1 = (1, 2, 2), orthogonal and both of length 3. Row 1: the
        # axis-aligned plane z = 1, whose unit edges are orthogonal too. Row 2: a general triangle.
        p0 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.1, 0.2, 0.3]], device=device, dtype=dtype)
        p1 = torch.tensor([[1.0, 2.0, 2.0], [1.0, 0.0, 1.0], [1.5, -0.4, 0.8]], device=device, dtype=dtype)
        p2 = torch.tensor([[2.0, 1.0, -2.0], [0.0, 1.0, 1.0], [-0.7, 1.1, 2.0]], device=device, dtype=dtype)
        for p in (p0, p1, p2):
            p.requires_grad_(True)
        plane = Hyperplane.through(p0, p1, p2)
        expected = torch.tensor([[4.0, -4.0, 2.0], [0.0, 0.0, -1.0]], device=device, dtype=dtype)
        self.assert_close(plane.normal.data[:2], expected / expected.norm(dim=-1, keepdim=True))
        # detect_anomaly also rejects a nan inside the backward that torch.where would discard, so this pins the
        # distinct singular values of the substituted matrix and not only the final gradient.
        with torch.autograd.detect_anomaly():
            (plane.normal.data.sum() + plane.offset.data.sum()).backward()
        for p in (p0, p1, p2):
            assert torch.isfinite(p.grad).all(), p.grad

    def test_through_orthogonal_equal_length_gradcheck_5056(self, device):
        # The three rows of the test above through gradcheck; the two orthogonal rows failed on main with nan.
        p0 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.1, 0.2, 0.3]], device=device, dtype=torch.float64)
        p1 = torch.tensor([[1.0, 2.0, 2.0], [1.0, 0.0, 1.0], [1.5, -0.4, 0.8]], device=device, dtype=torch.float64)
        p2 = torch.tensor([[2.0, 1.0, -2.0], [0.0, 1.0, 1.0], [-0.7, 1.1, 2.0]], device=device, dtype=torch.float64)

        def through(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
            plane = Hyperplane.through(p0, p1, p2)
            return torch.cat([plane.normal.data, plane.offset.data[..., None]], dim=-1)

        self.gradcheck(through, (p0, p1, p2))

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

    @pytest.mark.parametrize("scale", [1.0, 1e-4])
    def test_through_small_valid_triangle_keeps_orientation_5064(self, device, dtype, scale):
        # A small triangle is not collinear, and it keeps the (p2 - p0) x (p1 - p0) orientation. In float16 the
        # cross product of the 1e-4 triangle underflows to 0 (1e-8 is below the smallest subnormal), so it took the
        # SVD fallback and came back as +z (#5064); the cross product is now computed in float32.
        p0 = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([scale, 0.0, 0.0], device=device, dtype=dtype)
        p2 = torch.tensor([0.0, scale, 0.0], device=device, dtype=dtype)
        plane = Hyperplane.through(p0, p1, p2)
        self.assert_close(plane.normal.unwrap(), torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype))

    @pytest.mark.parametrize("size", [300.0, 4e4])
    def test_through_large_triangle_keeps_orientation_5064(self, device, dtype, size):
        # The normal is (p2 - p0) x (p1 - p0) = (size, size, 0) x (2 size, 0, 0) = (0, 0, -2 size^2). In float16 that
        # overflows for size 300 (-1.8e5 is beyond 65504), and for size 4e4 the edge p1 - p0 = 8e4 overflows too: the
        # first took the SVD fallback and came back as +z, the second as nan. Both are now computed in float32.
        p0 = torch.tensor([-size, 0.0, 1.0], device=device, dtype=dtype)
        p1 = torch.tensor([size, 0.0, 1.0], device=device, dtype=dtype)
        p2 = torch.tensor([0.0, size, 1.0], device=device, dtype=dtype)
        plane = Hyperplane.through(p0, p1, p2)
        self.assert_close(plane.normal.unwrap(), torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype))
        self.assert_close(plane.offset.data, torch.tensor(1.0, device=device, dtype=dtype))

    def test_through_points_collinear_up_to_rounding_raise(self, device, dtype):
        # The origin, (0.1, 0.2, 0.3) and (0.3, 0.6, 0.9) are collinear, but their rounded coordinates are not
        # exactly. The collinearity tolerance scales with the input dtype, also for a float16 or bfloat16 input whose
        # edges are taken in float32, so through() rejects these points in every dtype, as fit_plane does, instead
        # of returning a normal fitted to the rounding error ((0.89, -0.45, 0) in float16).
        points = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.3, 0.6, 0.9]], device=device, dtype=dtype)
        with pytest.raises(ValueCheckError, match="not collinear"):
            Hyperplane.through(points[0], points[1], points[2])
        with pytest.raises(ValueCheckError, match="not collinear"):
            fit_plane(points)

    def test_through_thin_triangle_keeps_orientation_without_checks(self, device, dtype):
        # Without the value checks, as under torch.compile, only the fallback threshold decides. For a float16 or
        # bfloat16 input it uses the float32 epsilon, so this triangle, whose cross product is 2^-11 of the product of
        # its edge lengths, below the float16 and bfloat16 epsilons, keeps the (p2 - p0) x (p1 - p0) orientation in
        # every dtype instead of taking the SVD fallback. Both point orders are checked: the normal follows the order.
        checks_were_enabled = are_checks_enabled()
        disable_checks()
        try:
            p0 = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
            p1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
            p2 = torch.tensor([0.5, 2.0**-12, 0.0], device=device, dtype=dtype)
            expected = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype)
            self.assert_close(Hyperplane.through(p0, p1, p2).normal.unwrap(), expected)
            self.assert_close(Hyperplane.through(p0, p2, p1).normal.unwrap(), -expected)
        finally:
            if checks_were_enabled:
                enable_checks()

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

    @pytest.mark.parametrize("shape", [None, (1,), (2, 1)])
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
        projected_distance = pl1.signed_distance(pl1.projection(p0)).data
        assert projected_distance.shape == (shape or ())
        self.assert_close(projected_distance, torch.zeros_like(projected_distance))
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

    def test_batched_projection_preserves_shape_and_values(self, device, dtype):
        normal = Vector3(torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], device=device, dtype=dtype))
        anchor = Vector3(torch.tensor([[[2.0, 0.0, 0.0]], [[0.0, 3.0, 0.0]]], device=device, dtype=dtype))
        point = Vector3(torch.tensor([[[5.0, 4.0, 1.0]], [[6.0, 7.0, 2.0]]], device=device, dtype=dtype))
        plane = Hyperplane.from_vector(normal, anchor)
        projected = plane.projection(point)
        expected = torch.stack(
            [
                Hyperplane.from_vector(Vector3(normal.data[i, 0]), Vector3(anchor.data[i, 0]))
                .projection(Vector3(point.data[i, 0]))
                .data
                for i in range(2)
            ]
        )[:, None, :]
        assert projected.data.shape == point.data.shape
        self.assert_close(projected.data, expected)
        self.assert_close(plane.signed_distance(projected).data, torch.zeros(2, 1, device=device, dtype=dtype))

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

    def test_gradcheck(self, device):
        # A tilted triangle whose edges are neither orthogonal nor of equal length, so the SVD fallback
        # (which torch.where differentiates as well) has distinct singular values.
        p0 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float64)
        p1 = torch.tensor([0.0, 2.0, 0.0], device=device, dtype=torch.float64)
        p2 = torch.tensor([0.0, 0.0, 3.0], device=device, dtype=torch.float64)

        def proxy(a, b, c):
            plane = Hyperplane.through(a, b, c)
            return plane.normal.data, plane.offset.data

        self.gradcheck(proxy, (p0, p1, p2))

    def test_through_tilted_plane_unit_normal(self, device, dtype):
        # #5012: a tilted plane (no zero component in its normal) used to get the cross product divided by
        # its p=-1 "norm", here (-6, -3, -2) with length 7, so distances came out 7x too large.
        p0 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([0.0, 2.0, 0.0], device=device, dtype=dtype)
        p2 = torch.tensor([0.0, 0.0, 3.0], device=device, dtype=dtype)

        plane = Hyperplane.through(p0, p1, p2)

        # The unit normal (p2 - p0) x (p1 - p0) / 7 and the offset -p0 . n = 6 / 7.
        expected_normal = torch.tensor([-6.0, -3.0, -2.0], device=device, dtype=dtype) / 7.0
        self.assert_close(plane.normal.data, expected_normal)
        self.assert_close(plane.offset.data, torch.tensor(6.0 / 7.0, device=device, dtype=dtype))

        # The point x = (1, 2, 3) has signed distance -12 / 7 and projects to x + (12 / 7) n.
        x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        self.assert_close(plane.signed_distance(x).data, torch.tensor(-12.0 / 7.0, device=device, dtype=dtype))
        expected_projection = torch.tensor([-23.0, 62.0, 123.0], device=device, dtype=dtype) / 49.0
        # low_tolerance: the first coordinate is 1 - 72 / 49, which cancels in float16 and bfloat16.
        self.assert_close(plane.projection(x).data, expected_projection, low_tolerance=True)

    def test_through_axis_aligned_follows_cross_product_orientation(self, device, dtype):
        # An axis-aligned plane has a zero component in its cross product, so its p=-1 "norm" was 0 and it took
        # the SVD fallback, whose sign is arbitrary: for this plane z = 1 it returned (0, 0, 1). It now takes the
        # cross-product branch like every other plane, and the normal is (p2 - p0) x (p1 - p0) = (0, 0, -1).
        p0 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        p1 = torch.tensor([1.0, 0.0, 1.0], device=device, dtype=dtype)
        p2 = torch.tensor([0.0, 1.0, 1.0], device=device, dtype=dtype)
        plane = Hyperplane.through(p0, p1, p2)
        self.assert_close(plane.normal.data, torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype))
        origin = torch.zeros(3, device=device, dtype=dtype)
        self.assert_close(plane.signed_distance(origin).data, torch.tensor(1.0, device=device, dtype=dtype))

    def test_through_batched_rows_are_independent(self, device, dtype):
        # The collinearity threshold is per row. The p=-1 "norm" reduced over the whole batch, so an axis-aligned
        # row sent a tilted row to the SVD fallback too, where it got a different sign than on its own.
        p0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        p1 = torch.tensor([[0.0, 2.0, 0.0], [1.0, 0.0, 1.0]], device=device, dtype=dtype)
        p2 = torch.tensor([[0.0, 0.0, 3.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)
        batched = Hyperplane.through(p0, p1, p2)
        expected = torch.tensor([[-6.0 / 7.0, -3.0 / 7.0, -2.0 / 7.0], [0.0, 0.0, -1.0]], device=device, dtype=dtype)
        self.assert_close(batched.normal.data, expected)
        for i in range(2):
            self.assert_close(batched.normal.data[i], Hyperplane.through(p0[i], p1[i], p2[i]).normal.data)

    def test_through_batched_threshold_is_per_row(self, device, dtype):
        # The fallback threshold compares each row's cross product with that row's own edge lengths. Measured
        # against the whole batch, the 1e-4 triangle would fall below it next to the 1e4 one and take the SVD
        # fallback, whose sign differs from the cross product's for this triangle. The 1e4 triangle's cross product
        # overflows float16, so half-precision points are taken to float32 first (#5064).
        base = [
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
            torch.tensor([0.0, 2.0, 0.0], device=device, dtype=dtype),
            torch.tensor([0.0, 0.0, 3.0], device=device, dtype=dtype),
        ]
        batched = Hyperplane.through(*(torch.stack([1e-4 * p, 1e4 * p]) for p in base))
        expected = torch.tensor([-6.0, -3.0, -2.0], device=device, dtype=dtype) / 7.0
        self.assert_close(batched.normal.data, expected.expand(2, 3))

    def test_through_degenerate_takes_svd_fallback(self, device, dtype):
        # Collinear or coincident points leave a zero cross product, and the SVD fallback returns a finite unit normal.
        # The gradient is not pinned: the plane through collinear points is not unique, so the fallback normal is
        # not differentiable there. Checks are disabled so this also covers the fallback where eager calls reject
        # these inputs.
        checks_were_enabled = are_checks_enabled()
        disable_checks()
        try:
            a = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
            b = torch.tensor([2.0, 3.0, 4.0], device=device, dtype=dtype)
            collinear = [a, b, 2 * b - a]
            coincident = [a, a.clone(), a.clone()]
            for points in (collinear, coincident):
                plane = Hyperplane.through(*points)
                norm = torch.linalg.vector_norm(plane.normal.data, dim=-1)
                self.assert_close(norm, torch.tensor(1.0, device=device, dtype=dtype))

            # The fallback normal is the SVD null vector of the edges, so it is orthogonal to the line through the
            # collinear points. A unit norm alone does not pin that: (0, 0, 1) has one and is 3 off here.
            normal = Hyperplane.through(*collinear).normal.data
            zero = torch.tensor(0.0, device=device, dtype=dtype)
            self.assert_close((normal * (b - a)).sum(-1), zero, rtol=0.0, atol=0.1)

            # A degenerate row does not send the other rows of its batch to the fallback.
            tilted = [
                torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
                torch.tensor([0.0, 2.0, 0.0], device=device, dtype=dtype),
                torch.tensor([0.0, 0.0, 3.0], device=device, dtype=dtype),
            ]
            batched = Hyperplane.through(*(torch.stack([c, t]) for c, t in zip(collinear, tilted)))
            self.assert_close(batched.normal.data[1], Hyperplane.through(*tilted).normal.data)
        finally:
            if checks_were_enabled:
                enable_checks()


class TestConventionsHyperplane(BaseTester):
    def test_convention_hyperplane_offset_sign(self, device, dtype):
        # The plane is n . x + d = 0: from_vector(n, e) sets d = -n . e, and signed_distance(x) = n . x + d is positive
        # on the side the normal points to, negative on the other side, and equals d at the origin. With a unit
        # normal it is the Euclidean distance. Neither the constructor nor from_vector normalises the normal, so a
        # normal of length 3 triples the offset and signed_distance.
        normal = torch.tensor([2.0, 1.0, -2.0], device=device, dtype=dtype) / 3
        assert (normal.abs() >= 0.1).all()  # a tilted plane: no normal component near 0
        e = torch.tensor([1.0, 2.0, 0.5], device=device, dtype=dtype)
        plane = Hyperplane.from_vector(Vector3(normal), Vector3(e))

        self.assert_close(plane.offset.data, torch.tensor(-1.0, device=device, dtype=dtype))  # -n . e
        self.assert_close(plane.signed_distance(e).data, torch.tensor(0.0, device=device, dtype=dtype))
        self.assert_close(plane.signed_distance(e + 0.7 * normal).data, torch.tensor(0.7, device=device, dtype=dtype))
        self.assert_close(plane.signed_distance(e - 0.4 * normal).data, torch.tensor(-0.4, device=device, dtype=dtype))
        self.assert_close(plane.abs_distance(e - 0.4 * normal).data, torch.tensor(0.4, device=device, dtype=dtype))
        self.assert_close(plane.signed_distance(torch.zeros(3, device=device, dtype=dtype)).data, plane.offset.data)

        scaled = Hyperplane.from_vector(Vector3(3 * normal), Vector3(e))
        built = Hyperplane(Vector3(3 * normal), scaled.offset)
        for p in (scaled, built):
            self.assert_close(p.normal.data, 3 * normal)
            self.assert_close(p.offset.data, torch.tensor(-3.0, device=device, dtype=dtype))
            self.assert_close(p.signed_distance(e + 0.7 * normal).data, torch.tensor(2.1, device=device, dtype=dtype))

    def test_convention_hyperplane_through_normal_orientation(self, device, dtype):
        # through(p0, p1, p2) takes its normal along c = (p2 - p0) x (p1 - p0), Eigen's order: the opposite of the
        # right-hand normal of the loop p0 -> p1 -> p2. Swapping two points flips the normal and a cyclic shift keeps
        # it. Only the direction is compared; the length of the returned normal is not part of this convention.
        p0 = torch.tensor([1.0, 0.0, 0.2], device=device, dtype=dtype)
        p1 = torch.tensor([0.1, 1.2, 0.0], device=device, dtype=dtype)
        p2 = torch.tensor([0.3, 0.0, 1.5], device=device, dtype=dtype)
        c = torch.tensor([-1.56, -1.31, -0.84], device=device, dtype=dtype)  # (p2 - p0) x (p1 - p0)
        assert (c.abs() >= 0.1).all()  # a tilted plane: no normal component near 0
        c = c / torch.linalg.vector_norm(c)

        def unit_normal(plane: Hyperplane) -> torch.Tensor:
            normal = plane.normal.data
            return normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)

        one = torch.tensor(1.0, device=device, dtype=dtype)
        plane = Hyperplane.through(p0, p1, p2)
        self.assert_close((unit_normal(plane) * c).sum(-1), one)
        self.assert_close((unit_normal(Hyperplane.through(p0, p2, p1)) * c).sum(-1), -one)
        self.assert_close((unit_normal(Hyperplane.through(p1, p2, p0)) * c).sum(-1), one)
        # The plane passes through the three points: d = -n . p0 for the returned normal.
        for p in (p0, p1, p2):
            self.assert_close(
                plane.signed_distance(p).data / torch.linalg.vector_norm(plane.normal.data), torch.zeros_like(one)
            )

    def test_wart_hyperplane_state_not_registered_4923(self, device, dtype):
        # Hyperplane keeps its normal and offset as Vector3 / Scalar wrappers outside the module state (#4923), so
        # state_dict() is empty and .to() leaves both in the original dtype. Registering them, as ParametrizedLine
        # does, flips both assertions.
        normal = torch.tensor([2.0, 1.0, -2.0], device=device, dtype=dtype) / 3
        e = torch.tensor([1.0, 2.0, 0.5], device=device, dtype=dtype)
        plane = Hyperplane.from_vector(Vector3(normal), Vector3(e))
        assert list(plane.state_dict()) == []

        other = torch.float16 if dtype == torch.float32 else torch.float32  # float64 is unavailable on MPS
        moved = plane.to(other)
        assert moved.normal.data.dtype == dtype
        assert moved.offset.data.dtype == dtype


class TestConventionsFitPlane(BaseTester):
    def test_convention_fit_plane_input_forms(self, device, dtype):
        # fit_plane takes a tensor or a Vector3 of shape (N, 3), or a batch (B, N, 3) that it fits row by row. The
        # normal is unit, its sign is the SVD's and unspecified, so normals are compared up to sign (and the offset
        # with them); the plane passes through the centroid. Other coordinate counts raise TypeError.
        true_normal = torch.tensor([0.36, -0.48, 0.8], device=device, dtype=dtype)
        assert (true_normal.abs() >= 0.1).all()  # a tilted plane: no normal component near 0
        # Six points near the plane through (1, -2, 0.5) with this normal (offsets up to 0.02 along it).
        points = torch.tensor(
            [
                [1.0036, -2.0048, 0.5080],
                [0.7129, -0.3310, 1.6056],
                [-0.3940, -3.1425, 0.4606],
                [3.0525, -0.9923, 0.1810],
                [-0.0298, 0.4593, 2.4265],
                [1.8415, -2.1787, 0.0204],
            ],
            device=device,
            dtype=dtype,
        )
        # A second plane for the batch: the same points with their coordinates permuted, normal (-0.48, 0.8, 0.36).
        permuted = points[:, [1, 2, 0]]

        def assert_same_plane(normal: torch.Tensor, offset: torch.Tensor, other: Hyperplane) -> None:
            sign = torch.sign((normal * other.normal.data).sum(-1))
            self.assert_close(normal, sign * other.normal.data)
            self.assert_close(offset, sign * other.offset.data)

        plane = fit_plane(points)
        assert plane.normal.shape == (3,)
        assert plane.offset.shape == ()
        self.assert_close(torch.linalg.vector_norm(plane.normal.data), torch.tensor(1.0, device=device, dtype=dtype))
        assert (plane.normal.data * true_normal).sum().abs().item() > 0.999
        centroid = points.mean(0)
        self.assert_close(plane.signed_distance(centroid).data, torch.tensor(0.0, device=device, dtype=dtype))

        from_vector3 = fit_plane(Vector3(points))
        assert_same_plane(from_vector3.normal.data, from_vector3.offset.data, plane)

        batch = fit_plane(torch.stack([points, permuted]))
        assert batch.normal.shape == (2, 3)
        assert batch.offset.shape == (2,)
        rows = [plane, fit_plane(permuted)]
        assert (rows[0].normal.data * rows[1].normal.data).sum().abs().item() < 0.5  # the two rows differ
        for i, row in enumerate(rows):
            assert_same_plane(batch.normal.data[i], batch.offset.data[i], row)

        for wrong in (points[:, :2], torch.cat([points, points[:, :1]], -1)):
            with pytest.raises(TypeError, match=r"vector must be \(\*, 3\)"):
                fit_plane(wrong)
