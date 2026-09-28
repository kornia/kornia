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

    def test_through_tilted_plane_unit_normal(self, device, dtype):
        # Tilted plane fixture with no zero components in the normal
        p0 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        p1 = torch.tensor([0.0, 2.0, 0.0], device=device, dtype=dtype)
        p2 = torch.tensor([0.0, 0.0, 3.0], device=device, dtype=dtype)

        plane = Hyperplane.through(p0, p1, p2)

        # 1. Normal must be strictly unit length (L2 norm == 1.0)
        normal_norm = torch.linalg.vector_norm(plane.normal.data, dim=-1)
        assert torch.allclose(normal_norm, torch.tensor(1.0, device=device, dtype=dtype))

        # 2. Point x = (1, 2, 3) must have true signed distance -12/7 ≈ -1.7142857
        x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        expected_distance = torch.tensor(-12.0 / 7.0, device=device, dtype=dtype)
        assert torch.allclose(plane.signed_distance(x).data.squeeze(), expected_distance, atol=1e-5, rtol=1e-5)

        # 3. Projection of x onto the plane must lie on the plane (signed distance == 0)
        proj = plane.projection(x)
        assert torch.allclose(plane.signed_distance(proj).data.squeeze(), torch.tensor(0.0, device=device, dtype=dtype), atol=1e-5, rtol=1e-5)