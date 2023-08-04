import pytest
import torch

from kornia.geometry.liegroup import Se2, So2
from kornia.geometry.vector import Vector2
from kornia.testing import BaseTester


class TestSe2(BaseTester):
    def _make_rand_data(self, device, dtype, input_shape):
        batch_size = input_shape[0]
        shape = input_shape[1:] if batch_size is None else input_shape
        return torch.rand(shape, device=device, dtype=dtype)

    def test_smoke(self, device, dtype):
        z = torch.rand((2,), dtype=torch.cfloat, device=device)
        so2 = So2(z)
        t = torch.rand((1, 2), device=device, dtype=dtype)
        s = Se2(so2, t)
        assert isinstance(s, Se2)
        assert isinstance(s.r, So2)
        self.assert_close(s.r.z.data, z)
        self.assert_close(s.t, t)

    @pytest.mark.parametrize("input_shape", [(1,), (2,), (5,), ()])
    def test_cardinality(self, device, dtype, input_shape):
        t_input_shape = (*input_shape, 2)
        z = torch.randn((*input_shape, 2), dtype=dtype, device=device)
        t = torch.randn(t_input_shape, dtype=dtype, device=device)
        s = Se2(So2(torch.complex(z[..., 0], z[..., 1])), t)
        theta = torch.rand((*input_shape, 3), dtype=dtype, device=device)
        assert s.so2.z.shape == input_shape
        assert s.t.shape == t_input_shape
        assert (s * s).so2.z.shape == input_shape
        assert (s * s).t.shape == t_input_shape
        assert s.exp(theta).so2.z.shape == input_shape
        assert s.exp(theta).t.shape == t_input_shape
        assert s.log().shape == (*input_shape, 3)
        if not any(input_shape):
            expected_hat_shape = (3, 3)
        else:
            expected_hat_shape = (input_shape[0], 3, 3)
        assert s.hat(theta).shape == expected_hat_shape
        assert s.inverse().so2.z.shape == input_shape
        assert s.inverse().t.shape == t_input_shape

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exception(self, device, dtype, batch_size):
        with pytest.raises(ValueError):
            r = So2.random(batch_size)
            t1 = torch.randn((batch_size, 1), dtype=dtype, device=device)
            t2 = torch.randn((batch_size, 3), dtype=dtype, device=device)
            Se2(r, t1)
            Se2(r, t2)
        with pytest.raises(ValueError):
            theta = torch.rand((batch_size, 2), dtype=dtype, device=device)
            Se2.exp(theta)
        with pytest.raises(ValueError):
            v = torch.rand((batch_size, 2), dtype=dtype, device=device)
            Se2.hat(v)
        with pytest.raises(ValueError):
            omega = torch.rand((4, 4), dtype=dtype, device=device)
            Se2.vee(omega)
        with pytest.raises(TypeError):
            Se2.identity(1, device, dtype) * [1.0, 2.0, 1.0]
        with pytest.raises(ValueError):
            theta = torch.rand((batch_size, 2), dtype=dtype, device=device)
            Se2.hat(theta)
        with pytest.raises(Exception):
            Se2.identity(batch_size=0)
        with pytest.raises(Exception):
            Se2.random(batch_size=0)
        with pytest.raises(Exception):
            x = torch.rand(5, dtype=dtype, device=device)
            y = torch.rand(3, dtype=dtype, device=device)
            Se2.trans(x, y)

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
    def test_init(self, device, dtype, batch_size):
        s1 = Se2.random(batch_size, device, dtype)
        s2 = Se2(s1.r, s1.t)
        assert isinstance(s2, Se2)
        self.assert_close(s1.r.z, s2.r.z)
        self.assert_close(s1.t, s2.t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        z = torch.rand(batch_size, dtype=torch.cfloat, device=device)
        t = torch.rand((batch_size, 2), device=device, dtype=dtype)
        s = Se2(So2(z), t)
        for i in range(batch_size):
            s1 = s[i]
            self.assert_close(s1.r.z, z[i])
            self.assert_close(s1.t, t[i])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul(self, device, dtype, batch_size):
        s1 = Se2.identity(batch_size, device, dtype)
        s2 = Se2.random(batch_size, device, dtype)
        s1_pose_s2 = s1 * s2
        s2_pose_s2 = s2 * s2.inverse()
        zeros_vec = torch.zeros(2, device=device, dtype=dtype)
        if batch_size is not None:
            zeros_vec = zeros_vec.repeat(batch_size, 1)
        so2_expected = So2.identity(batch_size, device, dtype)
        self.assert_close(s1_pose_s2.r.z, s2.r.z)
        self.assert_close(s1_pose_s2.t, s2.t)
        self.assert_close(s2_pose_s2.r.z.real, so2_expected.z.real)
        self.assert_close(s2_pose_s2.r.z.imag, so2_expected.z.imag)
        self.assert_close(s2_pose_s2.t, zeros_vec)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul_vector(self, device, dtype, batch_size):
        s1 = Se2.identity(batch_size, device, dtype)
        if batch_size is None:
            shape = ()
        else:
            shape = (batch_size,)
        s2 = Se2(So2.identity(batch_size, device, dtype), Vector2.random(shape, device, dtype))
        s1_pose_s2 = s1 * s2
        s2_pose_s2 = s2 * s2.inverse()
        zeros_vec = torch.zeros(2, device=device, dtype=dtype)
        if batch_size is not None:
            zeros_vec = zeros_vec.repeat(batch_size, 1)
        so2_expected = So2.identity(batch_size, device, dtype)
        self.assert_close(s1_pose_s2.r.z, s2.r.z)
        self.assert_close(s1_pose_s2.t, s2.t)
        self.assert_close(s2_pose_s2.r.z.real, so2_expected.z.real)
        self.assert_close(s2_pose_s2.r.z.imag, so2_expected.z.imag)
        self.assert_close(s2_pose_s2.t, zeros_vec)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        t = self._make_rand_data(device, dtype, (batch_size, 2))
        theta = torch.zeros(batch_size if batch_size is not None else (), device=device, dtype=dtype)
        z = torch.zeros((batch_size, 2) if batch_size is not None else (2,), device=device, dtype=dtype)
        s = Se2.exp(torch.cat((t, theta[..., None]), -1))
        self.assert_close(s.r.z, So2.exp(theta).z)
        self.assert_close(s.t, z)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_log(self, device, dtype, batch_size):
        t = self._make_rand_data(device, dtype, (batch_size, 2))
        s = Se2(So2.identity(batch_size, device, dtype), t)
        s.log()
        zero_vec = torch.zeros(3, device=device, dtype=dtype)
        if batch_size is not None:
            zero_vec = zero_vec.repeat(batch_size, 1)
        self.assert_close(s.log(), zero_vec)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        a = self._make_rand_data(device, dtype, (batch_size, 3))
        b = Se2.exp(a).log()
        self.assert_close(b, a, low_tolerance=True)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat(self, device, dtype, batch_size):
        v = self._make_rand_data(device, dtype, (batch_size, 2))
        theta = self._make_rand_data(device, dtype, (batch_size, 1))
        s_hat = Se2.hat(torch.cat((v, theta), -1))
        self.assert_close(v, s_hat[..., 2, 0:2])
        self.assert_close(s_hat[..., 0:2, 0:2].squeeze(), So2.hat(theta).squeeze())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_vee(self, device, dtype, batch_size):
        omega = self._make_rand_data(device, dtype, input_shape=(batch_size, 3, 3))
        v = Se2.vee(omega)
        self.assert_close(torch.stack((v[..., 0], v[..., 1]), -1), omega[..., 2, :2])
        self.assert_close(v[..., -1], omega[..., 0, 1])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat_vee(self, device, dtype, batch_size):
        a = self._make_rand_data(device, dtype, (batch_size, 3))
        omega_hat = Se2.hat(a)
        b = Se2.vee(omega_hat)
        self.assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_identity(self, device, dtype, batch_size):
        s = Se2.random(batch_size)
        s_pose_s = s * Se2.identity(batch_size)
        self.assert_close(s_pose_s.so2.z.real, s.so2.z.real)
        self.assert_close(s_pose_s.so2.z.imag, s.so2.z.imag)
        self.assert_close(s.t, s_pose_s.t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_matrix(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size,))
        t = self._make_rand_data(device, dtype, (batch_size, 2))
        s = So2.exp(theta)
        p1 = s * t
        p2 = s.matrix() @ t[..., None]
        self.assert_close(p1, p2.squeeze(-1))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_from_matrix(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size,))
        t = self._make_rand_data(device, dtype, (batch_size, 2))
        s = So2.exp(theta)
        p1 = s * t
        RT = torch.eye(3, device=device, dtype=dtype)
        if batch_size is not None:
            RT = RT.repeat(batch_size, 1, 1)
        RT[..., :2, :2] = s.matrix()
        p2 = Se2.from_matrix(RT) * t
        self.assert_close(p1, p2)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_inverse(self, device, batch_size, dtype):
        s = Se2.random(batch_size, device, dtype)
        s_in_in = s.inverse().inverse()
        self.assert_close(s_in_in.so2.z.real, s.so2.z.real)
        self.assert_close(s_in_in.so2.z.imag, s.so2.z.imag)
        self.assert_close(s_in_in.t, s.t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_random(self, device, dtype, batch_size):
        s = So2.random(batch_size=batch_size, device=device, dtype=dtype)
        t = self._make_rand_data(device, dtype, (batch_size, 2))
        se2 = Se2(s, t)
        se2_in_se2 = se2.inverse() * se2
        i = Se2.identity(batch_size=batch_size, device=device, dtype=dtype)
        self.assert_close(se2_in_se2.so2.z.real, i.so2.z.real)
        self.assert_close(se2_in_se2.so2.z.imag, i.so2.z.imag)
        self.assert_close(se2_in_se2.t, i.t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans(self, device, dtype, batch_size):
        trans = self._make_rand_data(device, dtype, (batch_size, 2))
        x, y = trans[..., 0], trans[..., 1]
        se2 = Se2.trans(x, y)
        self.assert_close(se2.t, trans)
        self.assert_close(se2.so2.matrix(), So2.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans_x(self, device, dtype, batch_size):
        x = self._make_rand_data(device, dtype, (batch_size, 1)).squeeze(-1)
        zs = torch.zeros_like(x)
        se2 = Se2.trans_x(x)
        self.assert_close(se2.t, torch.stack((x, zs), -1))
        self.assert_close(se2.so2.matrix(), So2.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans_y(self, device, dtype, batch_size):
        y = self._make_rand_data(device, dtype, (batch_size, 1)).squeeze(-1)
        zs = torch.zeros_like(y)
        se2 = Se2.trans_y(y)
        self.assert_close(se2.t, torch.stack((zs, y), -1))
        self.assert_close(se2.so2.matrix(), So2.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_adjoint(self, device, dtype, batch_size):
        x = Se2.random(batch_size)
        y = Se2.random(batch_size)
        self.assert_close(x.inverse().adjoint(), x.adjoint().inverse())
        self.assert_close((x * y).adjoint(), x.adjoint() @ y.adjoint())
