import pytest
import torch

from kornia.geometry.liegroup import So2
from kornia.testing import BaseTester


class TestSo2(BaseTester):
    def _make_rand_data(self, device, dtype, input_shape):
        batch_size = input_shape[0]
        shape = input_shape[1:] if batch_size is None else input_shape
        return torch.rand(shape, device=device, dtype=dtype)

    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_smoke(self, device, cdtype):
        z = torch.randn(2, 1, dtype=cdtype, device=device)
        s = So2(z)
        assert isinstance(s, So2)
        self.assert_close(s.z.data, z.data)

    @pytest.mark.parametrize("input_shape", [(1,), (2,), (5,), ()])
    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_cardinality(self, device, dtype, input_shape, cdtype):
        z = torch.randn(input_shape, dtype=cdtype, device=device)
        s = So2(z)
        theta = torch.rand(input_shape, dtype=dtype, device=device)
        assert s.z.shape == input_shape
        assert (s * s).z.shape == input_shape
        assert s.exp(theta).z.shape == input_shape
        assert s.log().shape == input_shape
        if not any(input_shape):
            expected_hat_shape = (2, 2)
        else:
            expected_hat_shape = (input_shape[0], 2, 2)
        assert s.hat(theta).shape == expected_hat_shape
        assert s.inverse().z.shape == input_shape

    @pytest.mark.parametrize("input_shape", [(1, 2, 2), (2, 2, 2), (5, 2, 2), (2, 2)])
    def test_matrix_cardinality(self, device, dtype, input_shape):
        matrix = torch.rand(input_shape, dtype=dtype, device=device)
        s = So2.from_matrix(matrix)
        assert s.matrix().shape == input_shape

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_exception(self, batch_size, device, dtype, cdtype):
        with pytest.raises(ValueError):
            z = torch.randn(batch_size, 2, dtype=cdtype, device=device)
            assert So2(z)
        with pytest.raises(TypeError):
            assert So2.identity(1, device, dtype) * [1.0, 2.0, 1.0]
        with pytest.raises(ValueError):
            theta = torch.rand((2, 2), dtype=dtype, device=device)
            assert So2.exp(theta)
        with pytest.raises(ValueError):
            theta = torch.rand((2, 2), dtype=dtype, device=device)
            assert So2.hat(theta)
        with pytest.raises(ValueError):
            m = torch.rand((2, 2, 1), dtype=dtype, device=device)
            assert So2.from_matrix(m)
        with pytest.raises(Exception):
            assert So2.identity(batch_size=0)

    # TODO: implement me
    def test_gradcheck(self, device):
        pass

    # TODO: implement me
    def test_jit(self, device, dtype):
        pass

    # TODO: implement me
    def test_module(self, device, dtype):
        pass

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_init(self, device, dtype, batch_size, cdtype):
        z1 = self._make_rand_data(device, cdtype, (batch_size,))
        z2 = self._make_rand_data(device, cdtype, (batch_size, 1))
        z3_real = self._make_rand_data(device, dtype, (batch_size,))
        z3_imag = self._make_rand_data(device, dtype, (batch_size,))
        z3 = torch.complex(z3_real, z3_imag)
        s1 = So2(z1)
        s2 = So2(s1.z)
        assert isinstance(s2, So2)
        self.assert_close(s1.z, s2.z)
        self.assert_close(So2(z1).z, z1)
        self.assert_close(So2(z2).z, z2)
        self.assert_close(So2(z3).z, z3)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_getitem(self, device, batch_size, cdtype):
        z = self._make_rand_data(device, cdtype, (batch_size,))
        s = So2(z)
        n = 1 if batch_size is None else batch_size
        for i in range(n):
            if batch_size is None:
                expected = s.z
                actual = z
            else:
                expected = s[i].z.data.squeeze()
                actual = z[i]
            self.assert_close(expected, actual)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul(self, device, dtype, batch_size):
        s1 = So2.identity(batch_size, device, dtype)
        z = self._make_rand_data(device, dtype, (batch_size, 2))
        s2 = So2(torch.complex(z[..., 0], z[..., 1]))
        t1 = self._make_rand_data(device, dtype, (batch_size, 2))
        t2 = self._make_rand_data(device, dtype, (2,))
        s1_pose_s2 = s1 * s2
        s2_pose_s2 = s2 * s2.inverse()
        self.assert_close(s1_pose_s2.z.real, s2.z.real)
        self.assert_close(s1_pose_s2.z.imag, s2.z.imag)
        self.assert_close(s2_pose_s2.z.real, s1.z.real)
        self.assert_close(s2_pose_s2.z.imag, s1.z.imag)
        self.assert_close((s1 * t1), t1)
        self.assert_close((So2.identity(device=device, dtype=dtype) * t2), t2)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size, 1))
        s = So2.exp(theta)
        self.assert_close(s.z.real, theta.cos())
        self.assert_close(s.z.imag, theta.sin())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_log(self, device, batch_size, cdtype):
        z = self._make_rand_data(device, cdtype, (batch_size,))
        t = So2(z).log()
        self.assert_close(t, z.imag.atan2(z.real))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size, 1))
        self.assert_close(So2.exp(theta).log(), theta)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size,))
        m = So2.hat(theta)
        o = torch.ones((2, 1), device=device, dtype=dtype)
        self.assert_close((m @ o).reshape(-1, 2, 1), theta.reshape(-1, 1, 1).repeat(1, 2, 1))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_vee(self, device, dtype, batch_size):
        omega = self._make_rand_data(device, dtype, (batch_size, 2, 2))
        theta = So2.vee(omega)
        self.assert_close(omega[..., 0, 1], theta)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat_vee(self, device, dtype, batch_size):
        a = self._make_rand_data(device, dtype, (batch_size,))
        omega = So2.hat(a)
        b = So2.vee(omega)
        self.assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_matrix(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size,))
        t = self._make_rand_data(device, dtype, (batch_size, 2))
        s = So2.exp(theta)
        p1 = s * t
        p2 = s.matrix() @ t[..., None]
        self.assert_close(p1, p2.squeeze(-1))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_inverse(self, device, batch_size, cdtype):
        z = self._make_rand_data(device, cdtype, (batch_size,))
        s = So2(z)
        s_in_in = s.inverse().inverse()
        self.assert_close(s_in_in.z.real, z.real)
        self.assert_close(s_in_in.z.imag, z.imag)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_random(self, device, dtype, batch_size):
        s = So2.random(batch_size=batch_size, device=device, dtype=dtype)
        s_in_s = s.inverse() * s
        i = So2.identity(batch_size=batch_size, device=device, dtype=dtype)
        self.assert_close(s_in_s.z.real, i.z.real)
        self.assert_close(s_in_s.z.imag, i.z.imag)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_adjoint(self, device, dtype, batch_size):
        s = So2.identity(batch_size, device=device, dtype=dtype)
        self.assert_close(s.matrix(), s.adjoint())
