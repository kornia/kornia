import pytest

from kornia.geometry.plane import Hyperplane
from kornia.geometry.vector import _VectorType
from kornia.testing import BaseTester


class TestHyperplane(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("dim", [2, 3])
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
