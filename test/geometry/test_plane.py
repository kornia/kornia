import pytest
import torch

from kornia.geometry.plane import Hyperplane, fit_plane
from kornia.geometry.vector import _VectorType
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
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_smoke(self, device, dtype, batch_size, dim):
        p0 = _VectorType.random((batch_size, dim), device, dtype)
        n0 = _VectorType.random((batch_size, dim), device, dtype).normalized()
        pl0 = Hyperplane.from_vector(n0.data, p0.data)
        # TODO: improve api so that we can accept Vector too
        assert pl0.offset.shape == (batch_size,)
        assert pl0.normal.shape == (batch_size, dim)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_through_two(self, device, dtype, batch_size):
        v0 = _VectorType.random((batch_size, 2), device, dtype)
        v1 = _VectorType.random((batch_size, 2), device, dtype)
        # TODO: improve api so that we can accept Vector too
        p0 = Hyperplane.through(v0.data, v1.data)
        assert p0.offset.shape == (batch_size,)
        assert p0.normal.shape == (batch_size, 2)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_through_three(self, device, dtype, batch_size):
        v0 = _VectorType.random((batch_size, 3), device, dtype)
        v1 = _VectorType.random((batch_size, 3), device, dtype)
        v2 = _VectorType.random((batch_size, 3), device, dtype)
        # TODO: improve api so that we can accept Vector too
        p0 = Hyperplane.through(v0.data, v1.data, v2.data)
        assert p0.offset.shape == (batch_size,)
        assert p0.normal.shape == (batch_size, 3)

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
