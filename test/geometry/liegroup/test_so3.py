import pytest
import torch

from kornia.geometry.liegroup import So3
from kornia.geometry.quaternion import Quaternion
from kornia.testing import BaseTester


class TestSo3(BaseTester):
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

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_init(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size)
        q = q.to(device, dtype)
        s1 = So3(q)
        s2 = So3(s1.q)
        assert isinstance(s2, So3)
        self.assert_close(s1.q.data, s2.q.data)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
        for i in range(batch_size):
            s1 = s[i]
            self.assert_close(s1.q.data[0], q.data[i])

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_mul(self, device, dtype, batch_size):
        q1 = Quaternion.identity(batch_size)
        q1 = q1.to(device, dtype)
        q2 = Quaternion.random(batch_size)
        q2 = q2.to(device, dtype)
        t = torch.rand(batch_size, 3, device=device, dtype=dtype)
        s1 = So3(q1)
        s2 = So3(q2)
        self.assert_close((s1 * s2).q.data, s2.q.data)
        self.assert_close((s2 * s2.inverse()).q.data, s1.q.data)
        self.assert_close((s1 * t), t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_unit_norm(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size)
        q1 = q1.to(device, dtype)
        q2 = Quaternion.random(batch_size)
        q2 = q2.to(device, dtype)
        s1 = So3(q1)
        s2 = So3(q2)
        s3 = s1 * s2
        s4 = s1.inverse()
        s5 = s2.inverse()
        s6 = s3.inverse()
        for i in range(batch_size):
            self.assert_close(s1[i].q.norm(), torch.Tensor([[1.0]]))
            self.assert_close(s2[i].q.norm(), torch.tensor([[1.0]], device=device, dtype=dtype))
            self.assert_close(s3[i].q.norm(), torch.Tensor([[1.0]]))
            self.assert_close(s4[i].q.norm(), torch.Tensor([[1.0]]))
            self.assert_close(s5[i].q.norm(), torch.Tensor([[1.0]]))
            self.assert_close(s6[i].q.norm(), torch.Tensor([[1.0]]))

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
        zero_vec = torch.zeros((batch_size, 3), device=device, dtype=dtype)
        self.assert_close(s.exp(zero_vec).q.data, q.data)  # exp of zero vec is identity

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_log(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
        zero_vec = torch.zeros((batch_size, 3), device=device, dtype=dtype)
        self.assert_close(s.log(), zero_vec)  # log of identity quat is zero vec

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
        a = torch.rand(batch_size, 3, device=device, dtype=dtype)
        b = s.exp(a).log()
        self.assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_hat(self, device, dtype, batch_size):
        v = torch.tensor([1, 2, 3]).repeat(batch_size, 1)
        v = v.to(device, dtype)
        self.assert_close(So3.hat(v).unique()[-3:], v[0, :])

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_vee(self, device, dtype, batch_size):
        omega = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).repeat(batch_size, 1, 1)
        omega = omega.to(device, dtype)
        expected = torch.tensor([[8, 3, 4]]).repeat(batch_size, 1)
        expected = expected.to(device, dtype)
        self.assert_close(So3.vee(omega), expected)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_hat_vee(self, device, dtype, batch_size):
        a = torch.rand(batch_size, 3, device=device, dtype=dtype)
        omega = So3.hat(a)
        b = So3.vee(omega)
        self.assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_matrix(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        r = So3(q).matrix()
        for i in range(batch_size):
            q1 = q[i]
            r1 = r[i, :, :]
            pvec = torch.rand(3, device=device, dtype=dtype)
            pquat = Quaternion(torch.cat([torch.tensor([0]).to(device, dtype), pvec])[None, :])
            qp_ = q1 * pquat * q1.inv()
            rp_ = torch.matmul(r1, pvec)[None, :]
            self.assert_close(rp_, qp_.vec)  # p_ = R*p = q*p*q_inv
            self.assert_close(rp_.norm(), pvec.norm())

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_ortho(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        b_R_a = So3(q).matrix()
        a_R_b = So3(q).inverse().matrix()
        a_R_a = (So3(q) * So3(q).inverse()).matrix()

        self.assert_close(a_R_a, torch.eye(3, device=device, dtype=dtype).repeat(batch_size, 1, 1))

        for i in range(batch_size):
            self.assert_close(a_R_a[i, :, :], torch.eye(3))
            self.assert_close(a_R_b[i, :, :] @ b_R_a[i, :, :], torch.eye(3))
            self.assert_close(b_R_a[i, :, :] @ a_R_b[i, :, :], torch.eye(3))

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_inverse(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        self.assert_close(So3(q).inverse().inverse().q.data, q.data)
        self.assert_close(So3(q).inverse().inverse().matrix(), So3(q).matrix())
