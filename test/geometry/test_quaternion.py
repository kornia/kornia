import pytest
import torch

from kornia.geometry.quaternion import Quaternion
from kornia.testing import assert_close


class TestQuaternion:
    def assert_close(self, actual, expected, rtol=None, atol=None):
        if isinstance(actual, Quaternion):
            actual = actual.data.data
        elif isinstance(actual, torch.nn.Parameter):
            actual = actual.data
        if isinstance(expected, Quaternion):
            expected = expected.data.data
        elif isinstance(expected, torch.nn.Parameter):
            expected = expected.data

        assert_close(actual, expected, rtol=rtol, atol=atol)

    def test_smoke(self, device, dtype):
        q = Quaternion.from_coeffs(1.0, 0.0, 0.0, 0.0)
        q = q.to(device, dtype)
        q_data = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        assert isinstance(q, Quaternion)
        assert q.shape == (1, 4)
        self.assert_close(q.data, q_data)
        self.assert_close(q.q, q_data)
        self.assert_close(q.real, q_data[..., :1])
        self.assert_close(q.scalar, q_data[..., :1])
        self.assert_close(q.vec, q_data[..., 1:])

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_init(self, device, dtype, batch_size):
        q1 = Quaternion.identity(batch_size)
        q1 = q1.to(device, dtype)
        q2 = Quaternion(q1.data)
        assert isinstance(q2, Quaternion)
        self.assert_close(q1, q2)

    def test_init_fail(self, device, dtype):
        with pytest.raises(Exception):
            _ = Quaternion("q")

        with pytest.raises(Exception):
            _ = Quaternion([1, 0, 0, 0])

        with pytest.raises(Exception):
            _ = Quaternion(1, [0, 0, 0])

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_random(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        q_n = q.normalize().norm()
        self.assert_close(q_n, q_n.new_ones(q_n.shape))

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_minus(self, device, dtype, batch_size):
        data = torch.rand(batch_size, 4, device=device, dtype=dtype)
        q = Quaternion(data)
        q = q.to(device, dtype)
        self.assert_close(-q, -data)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_add(self, device, dtype, batch_size):
        d1 = torch.rand(batch_size, 4, device=device, dtype=dtype)
        d2 = torch.rand(batch_size, 4, device=device, dtype=dtype)
        q1 = Quaternion(d1).to(device, dtype)
        q2 = Quaternion(d2).to(device, dtype)
        q3 = q1 + q2
        assert isinstance(q3, Quaternion)
        self.assert_close(q3, d1 + d2)
        q1 += q2
        self.assert_close(q1, q3)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_subtract(self, device, dtype, batch_size):
        d1 = torch.rand(batch_size, 4, device=device, dtype=dtype)
        d2 = torch.rand(batch_size, 4, device=device, dtype=dtype)
        q1 = Quaternion(d1).to(device, dtype)
        q2 = Quaternion(d2).to(device, dtype)
        q3 = q1 - q2
        assert isinstance(q3, Quaternion)
        self.assert_close(q3, d1 - d2)
        q1 -= q2
        self.assert_close(q1, q3)

    def test_multiplication_of_bases(self, device, dtype):
        one = Quaternion.from_coeffs(1.0, 0.0, 0.0, 0.0).to(device, dtype)
        i = Quaternion.from_coeffs(0.0, 1.0, 0.0, 0.0).to(device, dtype)
        j = Quaternion.from_coeffs(0.0, 0.0, 1.0, 0.0).to(device, dtype)
        k = Quaternion.from_coeffs(0.0, 0.0, 0.0, 1.0).to(device, dtype)

        self.assert_close(i * i, j * j)
        self.assert_close(j * j, k * k)
        self.assert_close(k * k, i * j * k)
        self.assert_close(i * j * k, -one)

        self.assert_close(i * j, k)
        self.assert_close(i * i, -one)
        self.assert_close(i * k, -j)
        self.assert_close(j * i, -k)
        self.assert_close(j * j, -one)
        self.assert_close(j * k, i)
        self.assert_close(k * i, j)
        self.assert_close(k * j, -i)
        self.assert_close(k * k, -one)
        self.assert_close(i * j * k, -one)

    def test_division_of_bases(self, device, dtype):
        one = Quaternion.from_coeffs(1.0, 0.0, 0.0, 0.0).to(device, dtype)
        i = Quaternion.from_coeffs(0.0, 1.0, 0.0, 0.0).to(device, dtype)
        j = Quaternion.from_coeffs(0.0, 0.0, 1.0, 0.0).to(device, dtype)
        k = Quaternion.from_coeffs(0.0, 0.0, 0.0, 1.0).to(device, dtype)

        self.assert_close(i / i, j / j)
        self.assert_close(j / j, k / k)
        self.assert_close(k / k, one)
        self.assert_close(k / -k, -one)

        self.assert_close(i / j, -k)
        self.assert_close(i / i, one)
        self.assert_close(i / k, j)
        self.assert_close(j / i, k)
        self.assert_close(j / j, one)
        self.assert_close(j / k, -i)
        self.assert_close(k / i, -j)
        self.assert_close(k / j, i)
        self.assert_close(k / k, one)
        self.assert_close(i / -j, k)

    def test_pow(self, device, dtype):
        pass

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_inverse(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size)
        q1 = q1.to(device, dtype)
        q2 = Quaternion.identity(batch_size)
        q2 = q2.to(device, dtype)
        self.assert_close(q1 * q1.inv(), q2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_conjugate(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size)
        q1 = q1.to(device, dtype)
        q2 = Quaternion.random(batch_size)
        q2 = q2.to(device, dtype)
        self.assert_close((q1 * q2).conj(), q2.conj() * q1.conj())

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_double_conjugate(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size)
        q1 = q1.to(device, dtype)
        self.assert_close(q1, q1.conj().conj())

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_norm(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size)
        q1 = q1.to(device, dtype)
        q2 = Quaternion.random(batch_size)
        q2 = q2.to(device, dtype)
        self.assert_close((q1 * q2).norm(), q1.norm() * q2.norm())

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_normalize(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size)
        q1 = q1.to(device, dtype)
        q1_n = q1.normalize().norm()
        self.assert_close(q1_n, q1_n.new_ones(q1_n.shape))

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_matrix(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size)
        q1 = q1.to(device, dtype)
        m1 = q1.matrix()
        q2 = Quaternion.from_matrix(m1)
        for (qq1, qq2) in zip(q1.data, q2.data):
            try:
                self.assert_close(qq1, qq2)
            except Exception:
                self.assert_close(qq1, -qq2)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        for i in range(batch_size):
            q1 = q[i]
            self.assert_close(q1.data[0], q.data[i])

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_axis_angle(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size)
        q1 = q1.to(device, dtype)
        angle = 2 * q1.scalar.arccos()
        axis = q1.vec / (angle / 2).sin()
        axis_angle = axis * angle
        q2 = Quaternion.from_axis_angle(axis_angle)
        q2 = q2.to(device, dtype)
        self.assert_close(q1, q2)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_slerp(self, device, dtype, batch_size):
        for axis in torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]):
            axis = axis.repeat(batch_size, 1)
            q1 = Quaternion.from_axis_angle(axis * 0)
            q1.to(device, dtype)
            q2 = Quaternion.from_axis_angle(axis * 3.14159)
            q2.to(device, dtype)
            for t in torch.linspace(0.1, 1, 10):
                q3 = q1.slerp(q2, t)  # zero theta case i.e dot product is one
                q4 = Quaternion.from_axis_angle(axis * t * 3.14159)
                self.assert_close(q3, q4)
