import pytest
import torch

from kornia.geometry.conversions import euler_from_quaternion
from kornia.geometry.liegroup import So3
from kornia.geometry.quaternion import Quaternion
from kornia.geometry.vector import Vector3
from kornia.testing import BaseTester


class TestSo3(BaseTester):
    def _make_rand_data(self, device, dtype, batch_size, dims):
        shape = [] if batch_size is None else [batch_size]
        return torch.rand(shape + [dims], device=device, dtype=dtype)

    def test_smoke(self, device, dtype):
        q = Quaternion.from_coeffs(1.0, 0.0, 0.0, 0.0)
        q = q.to(device, dtype)
        s = So3(q)
        assert isinstance(s, So3)
        self.assert_close(s.q.data, q.data)

    # TODO: implement me
    def test_cardinality(self, device, dtype):
        pass

    # TODO: implement me
    def test_exception(self, device, dtype):
        pass

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
        q = Quaternion.identity(batch_size, device, dtype)
        s1 = So3(q)
        s2 = So3(s1.q)
        assert isinstance(s2, So3)
        self.assert_close(s1.q.data, s2.q.data)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        s = So3(q)
        for i in range(batch_size):
            s1 = s[i]
            self.assert_close(s1.q.data, q.data[i])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul(self, device, dtype, batch_size):
        q1 = Quaternion.identity(batch_size, device, dtype)
        q2 = Quaternion.random(batch_size, device, dtype)
        t = self._make_rand_data(device, dtype, batch_size, dims=3)
        s1 = So3(q1)
        s2 = So3(q2)
        self.assert_close((s1 * s2).q.data, s2.q.data)
        self.assert_close((s2 * s2.inverse()).q.data, s1.q.data)
        self.assert_close((s1 * t), t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul_vec(self, device, dtype, batch_size):
        q1 = Quaternion.identity(batch_size, device, dtype)
        q2 = Quaternion.random(batch_size, device, dtype)
        if batch_size is None:
            shape = ()
        else:
            shape = (batch_size,)
        t = Vector3.random(shape, device, dtype)
        s1 = So3(q1)
        s2 = So3(q2)
        self.assert_close((s1 * s2).q.data, s2.q.data)
        self.assert_close((s2 * s2.inverse()).q.data, s1.q.data)
        self.assert_close((s1 * t), t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_unit_norm(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size, device, dtype)
        q2 = Quaternion.random(batch_size, device, dtype)
        s1 = So3(q1)
        s2 = So3(q2)
        s3 = s1 * s2
        s4 = s1.inverse()
        s5 = s2.inverse()
        s6 = s3.inverse()

        ones_vec = torch.tensor(1.0, device=device, dtype=dtype)
        if batch_size is None:
            self.assert_close(s1.q.norm(), ones_vec)
            return

        for i in range(batch_size):
            self.assert_close(s1[i].q.norm(), ones_vec)
            self.assert_close(s2[i].q.norm(), ones_vec)
            self.assert_close(s3[i].q.norm(), ones_vec)
            self.assert_close(s4[i].q.norm(), ones_vec)
            self.assert_close(s5[i].q.norm(), ones_vec)
            self.assert_close(s6[i].q.norm(), ones_vec)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size, device, dtype)
        s = So3(q)
        zero_vec = 0 * self._make_rand_data(device, dtype, batch_size, dims=3)
        self.assert_close(s.exp(zero_vec).q.data, q.data)  # exp of zero vec is identity

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_log(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size, device, dtype)
        s = So3(q)
        zero_vec = 0 * self._make_rand_data(device, dtype, batch_size, dims=3)
        self.assert_close(s.log(), zero_vec)  # log of identity quat is zero vec

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        s = So3(q)
        a = self._make_rand_data(device, dtype, batch_size, dims=3)
        b = s.exp(a).log()
        self.assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat(self, device, dtype, batch_size):
        v = torch.tensor([1, 2, 3], device=device, dtype=dtype)
        expected = v
        if batch_size is not None:
            v = v.repeat(batch_size, 1)
        hat = So3.hat(v)
        if batch_size is None:
            hat = hat[None]
        self.assert_close(hat.unique()[-3:], expected)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_vee(self, device, dtype, batch_size):
        omega = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device, dtype=dtype)
        expected = torch.tensor([8, 3, 4], device=device, dtype=dtype)
        if batch_size is not None:
            omega = omega.repeat(batch_size, 1, 1)
            expected = expected.repeat(batch_size, 1)
        self.assert_close(So3.vee(omega), expected)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat_vee(self, device, dtype, batch_size):
        a = self._make_rand_data(device, dtype, batch_size, dims=3)
        omega = So3.hat(a)
        b = So3.vee(omega)
        self.assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_matrix(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        r = So3(q).matrix()
        if batch_size is None:
            q = Quaternion(q.data[None])
            r = r[None]
        for i in range(r.shape[0]):
            q1 = q[i]
            r1 = r[i, :, :]
            pvec = torch.rand(3, device=device, dtype=dtype)
            pquat = Quaternion(torch.cat([torch.tensor([0], device=device, dtype=dtype), pvec]))
            qp_ = q1 * pquat * q1.inv()
            rp_ = torch.matmul(r1, pvec)
            self.assert_close(rp_, qp_.vec)  # p_ = R*p = q*p*q_inv
            self.assert_close(rp_.norm(), pvec.norm())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_ortho(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        b_R_a = So3(q).matrix()
        a_R_b = So3(q).inverse().matrix()
        a_R_a = (So3(q) * So3(q).inverse()).matrix()

        eye_mat = torch.eye(3, device=device, dtype=dtype)
        if batch_size is None:
            eye_mat = eye_mat[None]
            a_R_a = a_R_a[None]
            a_R_b = a_R_b[None]
            b_R_a = b_R_a[None]
        if batch_size is not None:
            eye_mat = eye_mat.repeat(batch_size, 1, 1)

        self.assert_close(a_R_a, eye_mat)

        for i in range(eye_mat.shape[0]):
            self.assert_close(a_R_a[i, :, :], eye_mat[i])
            self.assert_close(a_R_b[i, :, :] @ b_R_a[i, :, :], eye_mat[i])
            self.assert_close(b_R_a[i, :, :] @ a_R_b[i, :, :], eye_mat[i])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_inverse(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        self.assert_close(So3(q).inverse().inverse().q.data, q.data)
        self.assert_close(So3(q).inverse().inverse().matrix(), So3(q).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_rot_x(self, device, dtype, batch_size):
        x = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        so3 = So3.rot_x(x)
        roll, _, _ = euler_from_quaternion(*so3.q.coeffs)
        self.assert_close(x, roll)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_rot_y(self, device, dtype, batch_size):
        y = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        so3 = So3.rot_y(y)
        _, pitch, _ = euler_from_quaternion(*so3.q.coeffs)
        self.assert_close(y, pitch)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_rot_z(self, device, dtype, batch_size):
        z = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        so3 = So3.rot_z(z)
        _, _, yaw = euler_from_quaternion(*so3.q.coeffs)
        self.assert_close(z, yaw)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_adjoint(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size, device, dtype)
        q2 = Quaternion.random(batch_size, device, dtype)
        x = So3(q1)
        y = So3(q2)
        self.assert_close(x.inverse().adjoint(), x.adjoint().inverse())
        self.assert_close((x * y).adjoint(), x.adjoint() @ y.adjoint())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_random(self, device, dtype, batch_size):
        s = So3.random(batch_size=batch_size, device=device, dtype=dtype)
        s_in_s = s.inverse() * s
        i = So3.identity(batch_size=batch_size, device=device, dtype=dtype)
        self.assert_close(s_in_s.q.data, i.q.data)
