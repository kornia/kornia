import pytest
import torch

from kornia.geometry.liegroup.so3 import So3
from kornia.geometry.quaternion import Quaternion
from kornia.testing import assert_close


class TestSo3:
    def test_smoke(self, device, dtype):
        q = Quaternion.from_coeffs(1.0, 0.0, 0.0, 0.0)
        q = q.to(device, dtype)
        s = So3(q)
        assert isinstance(s, So3)
        assert_close(s.q.data, q.data)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_init(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size)
        q = q.to(device, dtype)
        s1 = So3(q)
        s2 = So3(s1.q)
        assert isinstance(s2, So3)
        assert_close(s1.q.data, s2.q.data)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
        for i in range(batch_size):
            s1 = s[i]
            assert_close(s1.q.data[0], q.data[i])

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
        zero_vec = torch.zeros((batch_size, 3))
        assert_close(s.exp(zero_vec).q.data, q.data)  # exp of zero vec is identity

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_log(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
        zero_vec = torch.zeros((batch_size, 3))
        assert_close(s.log(), zero_vec)  # log of identity quat is zero vec

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
        a = torch.rand(batch_size, 3, device=device, dtype=dtype)
        b = s.exp(a).log()
        assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_hat(self, device, dtype, batch_size):
        v = torch.Tensor([1, 2, 3]).repeat(batch_size, 1)
        v = v.to(device, dtype)
        assert_close(So3.hat(v).unique()[-3:], v[0, :])

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_vee(self, device, dtype, batch_size):
        omega = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).repeat(batch_size, 1, 1)
        omega = omega.to(device, dtype)
        expected = torch.tensor([[8, 3, 4]]).repeat(batch_size, 1)
        expected = expected.to(device, dtype)
        assert_close(So3.vee(omega), expected)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_hat_vee(self, device, dtype, batch_size):
        a = torch.rand(batch_size, 3, device=device, dtype=dtype)
        omega = So3.hat(a)
        b = So3.vee(omega)
        assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_matrix(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
        r = s.matrix()
        for i in range(batch_size):
            q1 = q[i]
            r1 = r[i, :, :]
            pvec = torch.rand(3)
            pquat = Quaternion(torch.cat([torch.Tensor([0]), pvec])[None, :])
            qp_ = q1 * pquat * q1.inv()
            rp_ = torch.matmul(r1, pvec.T)[None, :]
            assert_close(rp_, qp_.vec)  # p_ = R*p = q*p*q_inv
