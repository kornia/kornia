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
from kornia.core.exceptions import BaseError
from kornia.geometry.plane import Hyperplane, fit_plane
from kornia.geometry.vector import Vector3

from testing.base import BaseTester


# TODO: implement the rest of methods
class TestFitPlane(BaseTester):
    @pytest.mark.parametrize("N", (4, 10))
    @pytest.mark.parametrize("D", (3,))
    # @pytest.mark.parametrize("D", (2, 3, 4))
    def test_smoke(self, device, dtype, N, D):
        points = torch.ones(N, D, device=device, dtype=dtype)
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
        # fallback, whose sign differs from the cross product's for this triangle.
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the 1e4 triangle's cross product overflows float16")
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
