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
        t = torch.rand(1, 3, device=device, dtype=dtype)
        s = Se3(So3(q), t)
        assert isinstance(s, Se3)
        assert isinstance(s.r, So3)
        assert_close(s.r.q.data, q.data)
        assert_close(s.t, t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_init(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size)
        q = q.to(device, dtype)
        t = t = torch.rand(batch_size, 3, device=device, dtype=dtype)
        s1 = Se3(So3(q), t)
        s2 = Se3(s1.r, s1.t)
        assert isinstance(s2, Se3)
        assert_close(s1.r.q.data, s2.r.q.data)
        assert_close(s1.t, s2.t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        t = torch.rand(batch_size, 3, device=device, dtype=dtype)
        s = Se3(So3(q), t)
        for i in range(batch_size):
            s1 = s[i]
            assert_close(s1.r.q.data[0], q.data[i])
            assert_close(s1.t[0], t[i])

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        omega = torch.zeros((batch_size, 3), device=device, dtype=dtype)
        t = torch.rand(batch_size, 3, device=device, dtype=dtype)
        s = Se3.exp(torch.cat((t, omega), -1))
        assert_close(s.r.q.data, Quaternion.identity(batch_size).to(device, dtype).data)
        assert_close(s.t, t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_log(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size)
        q = q.to(device, dtype)
        t = torch.rand(batch_size, 3, device=device, dtype=dtype)
        s = Se3(So3(q), t)
        zero_vec = torch.zeros((batch_size, 3), device=device, dtype=dtype)
        assert_close(s.log(), torch.cat((t, zero_vec), -1))

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        a = torch.rand(batch_size, 6, device=device, dtype=dtype)
        b = Se3.exp(a).log()
        assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_hat(self, device, dtype, batch_size):
        t = torch.rand((batch_size,3))
        t = t.to(device, dtype)
        omega = torch.tensor([1, 2, 3]).repeat(batch_size, 1)
        omega = omega.to(device, dtype)
        omega_hat = Se3.hat(torch.cat((t, omega), 1))
        assert omega_hat.shape == torch.Size([batch_size, 4, 4])
        assert_close(omega_hat[:, 3, 1:], t)
        assert_close(omega_hat[:, 0:3, 1:].unique()[-3:], torch.tensor([1, 2, 3]))

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_vee(self, device, dtype, batch_size):
        omega_hat = torch.Tensor([[
                               [0, 9, 5, 1], 
                               [0, 10, 6,  2],
                               [0, 11, 7,  3],
                               [0, 12, 8,  4]]]).repeat(batch_size, 1, 1)
        omega_hat = omega_hat.to(device, dtype)
        expected = torch.tensor([[12.0,  8.0,  4.0,  7.0,  1.0, 10.0]]).repeat(batch_size, 1)
        expected = expected.to(device, dtype)
        assert_close(Se3.vee(omega_hat), expected)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_hat_vee(self, device, dtype, batch_size):
        a = torch.rand(batch_size, 6, device=device, dtype=dtype)
        omega_hat = Se3.hat(a)
        b = Se3.vee(omega_hat)
        assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_matrix(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        t = torch.rand((batch_size, 3))
        t = t.to(device, dtype)
        rot = So3(q)
        s = Se3(rot, t)
        assert_close(s.matrix()[..., 0:3, 1:], rot.matrix())
        assert_close(s.matrix()[..., 3, 1:], t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_inverse(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size)
        q = q.to(device, dtype)
        rot = So3(q)
        t = torch.rand((batch_size, 3))
        t = t.to(device, dtype)
        sinv = Se3(rot, t).inverse()
        assert_close(sinv.r.inverse().q.data, q.data)
        assert_close(sinv.t, -t)
