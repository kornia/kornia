import tempfile

import pytest
import torch

from kornia.geometry.plane import Hyperplane, fit_plane
from kornia.geometry.vector import Vector3
from kornia.testing import BaseTester


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
        assert pl0.normal.shape == shape or () + (3,)
        assert pl0.offset.shape == (shape + () if shape is not None else ())

    def test_serialization(self, device, dtype):
        p = Vector3.random((), device, dtype)
        n = Vector3.random((), device, dtype).normalized()
        plane = Hyperplane.from_vector(n, p)
        with tempfile.NamedTemporaryFile() as tmp:
            file_path = tmp.name + ".pt"
            torch.save(plane, file_path)
            loaded_plane = torch.load(file_path)
            self.assert_close(plane.normal.unwrap(), loaded_plane.normal.unwrap())

    # TODO: implement `Vector2`
    # @pytest.mark.parametrize("batch_size", [1, 2])
    # def test_through_two(self, device, dtype, batch_size):
    #    v0 = _VectorType.random((batch_size, 2), device, dtype)
    #    v1 = _VectorType.random((batch_size, 2), device, dtype)
    #    # TODO: improve api so that we can accept Vector too
    #    p0 = Hyperplane.through(v0.data, v1.data)
    #    assert p0.offset.shape == (batch_size,)
    #    assert p0.normal.shape == (batch_size, 2)

    @pytest.mark.parametrize("shape", (None, (1,), (2, 1)))
    def test_through_three(self, device, dtype, shape):
        v0 = Vector3.random(shape, device, dtype)
        v1 = Vector3.random(shape, device, dtype)
        v2 = Vector3.random(shape, device, dtype)
        # TODO: improve api so that we can accept Vector too
        p0 = Hyperplane.through(v0, v1, v2)
        assert p0.normal.shape == shape or () + (3,)
        assert p0.offset.shape == (shape + () if shape is not None else ())

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
