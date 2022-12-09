import pytest
import torch

from kornia.geometry.conversions import QuaternionCoeffOrder, euler_from_quaternion, rotation_matrix_to_quaternion
from kornia.geometry.liegroup import Se3, So3
from kornia.geometry.quaternion import Quaternion
from kornia.testing import BaseTester


class TestSe3(BaseTester):
    def _make_rand_se3d(self, device, dtype, batch_size) -> Se3:
        q = Quaternion.random(batch_size, device, dtype)
        t = self._make_rand_data(device, dtype, batch_size, dims=3)
        return Se3(So3(q), t)

    def _make_rand_data(self, device, dtype, batch_size, dims):
        shape = [1] if batch_size is None else [batch_size]
        return torch.rand(shape + [dims], device=device, dtype=dtype)

    def test_smoke(self, device, dtype):
        q = Quaternion.from_coeffs(1.0, 0.0, 0.0, 0.0)
        q = q.to(device, dtype)
        t = torch.rand(1, 3, device=device, dtype=dtype)
        s = Se3(So3(q), t)
        assert isinstance(s, Se3)
        assert isinstance(s.r, So3)
        self.assert_close(s.r.q.data, q.data)
        self.assert_close(s.t, t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_cardinality(self, device, dtype, batch_size):
        se: Se3 = self._make_rand_se3d(device, dtype, batch_size)
        assert se.r.q.shape[0] == batch_size

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
        s1: Se3 = self._make_rand_se3d(device, dtype, batch_size)
        s2 = Se3(s1.r, s1.t)
        assert isinstance(s2, Se3)
        self.assert_close(s1.r.q.data, s2.r.q.data)
        self.assert_close(s1.t, s2.t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        t = torch.rand(batch_size, 3, device=device, dtype=dtype)
        s = Se3(So3(q), t)
        for i in range(batch_size):
            s1 = s[i]
            self.assert_close(s1.r.q.data[0], q.data[i])
            self.assert_close(s1.t[0], t[i])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul(self, device, dtype, batch_size):
        s1 = Se3.identity(batch_size, device, dtype)
        s2: Se3 = self._make_rand_se3d(device, dtype, batch_size)
        s1s2 = s1 * s2
        s2s2inv = s2 * s2.inverse()
        zeros_vec = torch.zeros(3, device=device, dtype=dtype)
        batch_size = batch_size if batch_size is not None else 1
        zeros_vec = zeros_vec.repeat(batch_size, 1)
        so3_expected = So3.identity(batch_size, device, dtype)
        self.assert_close(s1s2.r.q.data, s2.r.q.data)
        self.assert_close(s1s2.t, s2.t)
        self.assert_close(s2s2inv.r.q.data, so3_expected.q.data)
        self.assert_close(s2s2inv.t, zeros_vec)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul_point(self, device, dtype, batch_size):
        world_pose_s1: Se3 = self._make_rand_se3d(device, dtype, batch_size)
        world_pose_s2: Se3 = self._make_rand_se3d(device, dtype, batch_size)
        pt_in_world = self._make_rand_data(device, dtype, batch_size, dims=3)
        s1_pose_s2: Se3 = world_pose_s1.inverse() * world_pose_s2
        pt_in_s1 = world_pose_s1.inverse() * pt_in_world
        pt_in_s2 = world_pose_s2.inverse() * pt_in_world
        pt_in_s1_in_s2 = s1_pose_s2.inverse() * pt_in_s1
        pt_in_s2_in_s1 = s1_pose_s2 * pt_in_s2
        self.assert_close(pt_in_s1, pt_in_s2_in_s1)
        self.assert_close(pt_in_s2, pt_in_s1_in_s2)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        omega = torch.zeros(3, device=device, dtype=dtype)
        t = torch.rand(3, device=device, dtype=dtype)
        batch_size = batch_size if batch_size is not None else 1
        omega = omega.repeat(batch_size, 1)
        t = t.repeat(batch_size, 1)
        s = Se3.exp(torch.cat((t, omega), -1))
        quat_expected = Quaternion.identity(batch_size, device, dtype)
        self.assert_close(s.r.q.data, quat_expected.data)
        self.assert_close(s.t, t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_log(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size, device, dtype)
        t = self._make_rand_data(device, dtype, batch_size, dims=3)
        s = Se3(So3(q), t)
        zero_vec = torch.zeros(3, device=device, dtype=dtype)
        batch_size = batch_size if batch_size is not None else 1
        zero_vec = zero_vec.repeat(batch_size, 1)
        self.assert_close(s.log(), torch.cat((t, zero_vec), -1))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        a = self._make_rand_data(device, dtype, batch_size, dims=6)
        b = Se3.exp(a).log()
        self.assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat_vee(self, device, dtype, batch_size):
        a = self._make_rand_data(device, dtype, batch_size, dims=6)
        omega_hat = Se3.hat(a)
        b = Se3.vee(omega_hat)
        self.assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_matrix(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        t = self._make_rand_data(device, dtype, batch_size, dims=3)
        rot = So3(q)
        s = Se3(rot, t)
        rot_mat = s.matrix()
        assert rot_mat.shape[-2:] == (4, 4)
        if batch_size is not None:
            assert rot_mat.shape[0] == batch_size
        self.assert_close(rot_mat[..., 0:3, 0:3], rot.matrix())
        self.assert_close(rot_mat[..., 0:3, 3], t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_inverse(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        rot = So3(q)
        t = self._make_rand_data(device, dtype, batch_size, dims=3)
        sinv = Se3(rot, t).inverse()
        self.assert_close(sinv.r.inverse().q.data, q.data)
        self.assert_close(sinv.t, sinv.r * (-1 * t))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_rot_x(self, device, dtype, batch_size):
        x = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        se3 = Se3.rot_x(x)
        quat = rotation_matrix_to_quaternion(se3.so3.matrix(), order=QuaternionCoeffOrder.WXYZ)
        quat = Quaternion(quat)
        roll, _, _ = euler_from_quaternion(*quat.coeffs)
        self.assert_close(x, roll)
        self.assert_close(se3.t, torch.zeros_like(se3.t))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_rot_y(self, device, dtype, batch_size):
        y = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        se3 = Se3.rot_y(y)
        quat = rotation_matrix_to_quaternion(se3.so3.matrix(), order=QuaternionCoeffOrder.WXYZ)
        quat = Quaternion(quat)
        _, pitch, _ = euler_from_quaternion(*quat.coeffs)
        self.assert_close(y, pitch)
        self.assert_close(se3.t, torch.zeros_like(se3.t))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_rot_z(self, device, dtype, batch_size):
        z = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        se3 = Se3.rot_z(z)
        quat = rotation_matrix_to_quaternion(se3.so3.matrix(), order=QuaternionCoeffOrder.WXYZ)
        quat = Quaternion(quat)
        _, _, yaw = euler_from_quaternion(*quat.coeffs)
        self.assert_close(z, yaw)
        self.assert_close(se3.t, torch.zeros_like(se3.t))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans(self, device, dtype, batch_size):
        trans = self._make_rand_data(device, dtype, batch_size, dims=3)
        x, y, z = trans[..., 0], trans[..., 1], trans[..., 2]
        se3 = Se3.trans(x, y, z)
        self.assert_close(se3.t, trans)
        self.assert_close(se3.so3.matrix(), So3.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans_x(self, device, dtype, batch_size):
        x = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        zs = torch.zeros_like(x)
        se3 = Se3.trans_x(x)
        self.assert_close(se3.t, torch.stack((x, zs, zs), -1))
        self.assert_close(se3.so3.matrix(), So3.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans_y(self, device, dtype, batch_size):
        y = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        zs = torch.zeros_like(y)
        se3 = Se3.trans_y(y)
        self.assert_close(se3.t, torch.stack((zs, y, zs), -1))
        self.assert_close(se3.so3.matrix(), So3.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans_z(self, device, dtype, batch_size):
        z = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        zs = torch.zeros_like(z)
        se3 = Se3.trans_z(z)
        self.assert_close(se3.t, torch.stack((zs, zs, z), -1))
        self.assert_close(se3.so3.matrix(), So3.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_adjoint(self, device, dtype, batch_size):
        x_data = self._make_rand_data(device, dtype, batch_size, dims=6)
        y_data = self._make_rand_data(device, dtype, batch_size, dims=6)
        x = Se3.exp(x_data)
        y = Se3.exp(y_data)
        self.assert_close(x.inverse().adjoint(), x.adjoint().inverse())
        self.assert_close((x * y).adjoint(), x.adjoint() @ y.adjoint())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_random(self, device, dtype, batch_size):
        s = Se3.random(batch_size=batch_size, device=device, dtype=dtype)
        s_in_s = s.inverse() * s
        i = Se3.identity(batch_size=batch_size, device=device, dtype=dtype)
        self.assert_close(s_in_s.so3.q.data, i.so3.q.data)
        self.assert_close(s_in_s.t, i.t)
