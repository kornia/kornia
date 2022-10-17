import pytest
import torch

from kornia.geometry.liegroup.so3 import So3
from kornia.geometry.quaternion import Quaternion
from kornia.geometry.liegroup.se3 import Se3
from kornia.testing import assert_close


class TestSe3:
    def test_smoke(self, device, dtype):
        q = Quaternion.from_coeffs(1.0, 0.0, 0.0, 0.0)
        q = q.to(device, dtype)
        t = torch.rand((1,3))
        s = Se3(So3(q), t)
        assert isinstance(s, Se3)
        assert isinstance(s.r, So3)
        assert_close(s.r.q.data, q.data)
        assert_close(s.t, t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_init(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size)
        q = q.to(device, dtype)
        t = torch.rand((batch_size,3))
        s1 = Se3(So3(q), t)
        s2 = Se3(s1.r, s1.t)
        assert isinstance(s2, Se3)
        assert_close(s1.r.q.data, s2.r.q.data)
        assert_close(s1.t, s2.t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        t = torch.rand((batch_size,3))
        s = Se3(So3(q), t)
        for i in range(batch_size):
            s1 = s[i]
            assert_close(s1.r.q.data[0], q.data[i])
            assert_close(s1.t[0], t[i])

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        omega = torch.zeros((batch_size, 3))
        t = torch.rand((batch_size,3))
        s = Se3.exp(torch.cat((t, omega), -1))
        assert_close(s.r.q.data, Quaternion.identity(batch_size).to(device, dtype).data)
        assert_close(s.t, t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_log(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size)
        q = q.to(device, dtype)
        t = torch.rand((batch_size,3))
        s = Se3(So3(q), t)
        zero_vec = torch.zeros((batch_size, 3))
        assert_close(s.log(), torch.cat((t, zero_vec), -1))  # log of identity quat is zero vec

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        t = torch.rand((batch_size,3))
        s = Se3(So3(q), t)
        a = torch.rand(batch_size, 6, device=device, dtype=dtype)
        b = s.exp(a).log()
        assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_hat(self, device, dtype, batch_size):
        t = torch.rand((batch_size,3))
        omega = torch.tensor([1, 2, 3]).repeat(batch_size, 1)
        omega_hat = Se3.hat(torch.cat((t, omega), 1))
        assert omega_hat.shape == torch.Size([batch_size, 4, 4])
        assert_close(omega_hat[:, 3, 1:4], t)
        assert_close(omega_hat[:, 0:3, 1:4].unique()[-3:], torch.tensor([1, 2, 3]))

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_vee(self, device, dtype, batch_size):
        omega = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).repeat(batch_size, 1, 1)
        omega = omega.to(device, dtype)
        expected = torch.tensor([[8, 3, 4]]).repeat(batch_size, 1)
        expected = expected.to(device, dtype)
        assert_close(So3.vee(omega), expected)

    # @pytest.mark.parametrize("batch_size", (1, 2, 5))
    # def test_hat_vee(self, device, dtype, batch_size):
    #     a = torch.rand(batch_size, 3, device=device, dtype=dtype)
    #     omega = So3.hat(a)
    #     b = So3.vee(omega)
    #     assert_close(b, a)

    # @pytest.mark.parametrize("batch_size", (1, 2, 5))
    # def test_matrix(self, device, dtype, batch_size):
    #     q = Quaternion.random(batch_size)
    #     q = q.to(device, dtype)
    #     s = So3(q)
    #     r = s.matrix()
    #     for i in range(batch_size):
    #         q1 = q[i]
    #         r1 = r[i, :, :]
    #         pvec = torch.rand(3)
    #         pquat = Quaternion(torch.cat([torch.Tensor([0]), pvec])[None, :])
    #         qp_ = q1 * pquat * q1.inv()
    #         rp_ = torch.matmul(r1, pvec.T)[None, :]
    #         assert_close(rp_, qp_.vec)  # p_ = R*p = q*p*q_inv
