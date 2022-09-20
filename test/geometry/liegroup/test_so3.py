import pytest
import torch

from kornia.geometry.liegroup.so3 import So3
from kornia.geometry.quaternion import Quaternion

from kornia.testing import assert_close


class TestSo3:
    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        s = So3(q)
        a = torch.rand(batch_size, 3, device=device, dtype=dtype)
        b = s.exp(a).log()
        assert_close(b, a)

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
        qq = Quaternion.random(batch_size)
        qq = qq.to(device, dtype)
        ss = So3(qq)
        rr = ss.matrix()
        pp = torch.rand(batch_size, 3, device=device, dtype=dtype)
        for i in range(batch_size):
            q = Quaternion(qq[i][None,:]) # change once quaternion index issue is solved
            r = rr[i,:,:]
            pvec = torch.rand((3))
            pquat = Quaternion(torch.cat([torch.Tensor([0]), pvec])[None,:])
            qp_ = q * pquat * q.inv()
            rp_ = torch.matmul(r, pvec.T)[None, :]
            assert_close(rp_, qp_.vec)


        