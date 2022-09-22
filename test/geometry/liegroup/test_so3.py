import pytest
import torch

from kornia.geometry.liegroup.so3 import So3
from kornia.geometry.quaternion import Quaternion

from kornia.testing import assert_close


class TestSo3:

    # @pytest.mark.parametrize("batch_size", (1, 2, 5))
    # def test_init(self, device, dtype, batch_size):
    #     raise NotImplementedError

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
        zero_vec = torch.zeros((batch_size, 3))
        assert_close(s.exp(zero_vec)[:], q[:])# exp of zero vec is identity

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_log(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
        zero_vec = torch.zeros((batch_size, 3))
        assert_close(s.log(), zero_vec)# log of identity quat is zero vec

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
        a = torch.rand(batch_size, 3, device=device, dtype=dtype)
        b = s.exp(a).log()
        assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_hat(self, batch_size):
        s = So3.identity(1)
        v = torch.Tensor([1, 2, 3]).repeat(batch_size, 1)
        assert_close(s.hat(v).unique()[-3:], v[0,:])

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_vee(self, device, dtype, batch_size):
        s = So3.identity(1)
        omega = torch.Tensor([[[1, 2, 3],[4, 5, 6],[7, 8, 9]]]).repeat(batch_size, 1, 1)
        expected = torch.tensor([[8, 3, 4]]).repeat(batch_size, 1)
        assert_close(s.vee(omega), expected)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_hat_vee(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
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
        p = torch.rand(batch_size, 3, device=device, dtype=dtype)
        for i in range(batch_size):
            q1 = Quaternion(q[i][None,:]) # possible Quaternion index bug?
            r1 = r[i,:,:]
            pvec = torch.rand((3))
            pquat = Quaternion(torch.cat([torch.Tensor([0]), pvec])[None,:])
            qp_ = q1 * pquat * q1.inv()
            rp_ = torch.matmul(r1, pvec.T)[None, :]
            assert_close(rp_, qp_.vec) #p_ = R*p = q*p*q_inv
