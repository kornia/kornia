import pytest
import torch

from kornia.geometry.plane import Hyperplane, fit_plane
from kornia.geometry.vector import Vector3
from kornia.testing import BaseTester


# TODO: implement the rest of methods
class TestFitPlane(BaseTester):
    @pytest.mark.parametrize("B", (1, 2))
    @pytest.mark.parametrize("D", (2, 3, 4))
    def test_smoke(self, device, dtype, B, D):
        N: int = 10  # num points
        points = torch.ones(B, N, D, device=device, dtype=dtype)
        plane = fit_plane(points)
        assert isinstance(plane, Hyperplane)
        assert plane.offset.shape == (B,)
        assert plane.normal.shape == (B, D)

        assert (plane.normal == plane[0]).all()
        assert (plane.offset == plane[1]).all()

        normal, offset = fit_plane(points)
        assert (plane.normal == normal).all()
        assert (plane.offset == offset).all()

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
        assert pl0.normal.shape == (shape + (3,) if shape is not None else (3,))
        assert pl0.offset.shape == (shape + () if shape is not None else ())

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
        assert p0.normal.shape == (shape + (3,) if shape is not None else (3,))
        assert p0.offset.shape == (shape + () if shape is not None else ())

    def test_signed_distance(self, device, dtype):
        v0 = Vector3.from_coords(1.0, 0.0, 0.0)
        v1 = Vector3.from_coords(0.0, 1.0, 0.0)
        v2 = Vector3.from_coords(0.0, 0.0, 1.0)
        plane_in_world = Hyperplane.through(v0, v1, v2)
        p_in_world = Vector3.from_coords(0.0, 0.0, 1.0)
        # TODO: this seems to be wrong
        p_in_plane = plane_in_world.projection(p_in_world)
        print(p_in_plane)
        p_dist = plane_in_world.signed_distance(p_in_plane)
        print(p_dist)

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
