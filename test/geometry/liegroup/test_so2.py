import pytest
import torch

from kornia.geometry.liegroup import So2
from kornia.testing import BaseTester


class TestSo2(BaseTester):
    def test_smoke(self, device, dtype):
        z = torch.randn(2, 1, dtype=torch.cfloat)
        s = So2(z)
        assert isinstance(s, So2)
        self.assert_close(s.z.data, z.data)

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
        z = torch.randn(batch_size, 1, dtype=torch.cfloat, device=device)
        s1 = So2(z)
        s2 = So2(s1.z)
        assert isinstance(s2, So2)
        self.assert_close(s1.z.data, s2.z.data)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        data = torch.rand(batch_size, 2, device=device, dtype=dtype)
        z = torch.complex(data[..., 0, None], data[..., 1, None])
        s = So2(z)
        for i in range(batch_size):
            s1 = s[i]
            self.assert_close(s1.z.data[0].real, z.data[i].real)
            self.assert_close(s1.z.data[0].imag, z.data[i].imag)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_mul(self, device, dtype, batch_size):
        s1 = So2.identity(batch_size, device, dtype)
        z = torch.rand(batch_size, 2, device=device, dtype=dtype)
        s2 = So2(torch.complex(z[..., 0, None], z[..., 1, None]))
        t = torch.rand((batch_size, 2, 1), device=device, dtype=dtype)
        s1_pose_s2 = s1 * s2
        s2_pose_s2 = s2 * s2.inverse() #TODO naming correct?
        self.assert_close(s1_pose_s2.z.real, s2.z.real)
        self.assert_close(s1_pose_s2.z.imag, s2.z.imag)
        self.assert_close(s2_pose_s2.z.real, s1.z.real)
        self.assert_close(s2_pose_s2.z.imag, s1.z.imag)
        self.assert_close((s1 * t), t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        pi = 3.1415
        theta = torch.tensor([0, pi / 4, pi / 2, 3 * pi / 4, pi] * batch_size, device=device, dtype=dtype)
        s = So2.exp(theta[..., None])
        self.assert_close(s.z.real, theta[..., None].cos())
        self.assert_close(s.z.imag, theta[..., None].sin())

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_log(self, device, dtype, batch_size):
        data = torch.rand(batch_size, 2, device=device, dtype=dtype)
        z = torch.complex(data[..., 0, None], data[..., 1, None])
        t = So2(z).log()
        self.assert_close(t, data[..., 1, None].atan2(data[..., 0, None]))

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        pi = 3.1415
        theta = torch.tensor([0, pi / 4, pi / 2, 3 * pi / 4, pi] * batch_size, device=device, dtype=dtype)
        self.assert_close(So2.exp(theta).log(), theta)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_hat(self, device, dtype, batch_size):
        pi = 3.1415
        theta = torch.tensor([0, pi / 4, pi / 2, 3 * pi / 4, pi] * batch_size, device=device, dtype=dtype)
        m = So2.hat(theta[..., None])
        o = torch.ones((1, 2, 1), device=device, dtype=dtype)
        self.assert_close(m @ o, torch.cat((theta[..., None, None], theta[..., None, None]), 1))

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_matrix(self, device, dtype, batch_size):
        pi = 3.1415
        theta = torch.tensor([pi / 2] * batch_size, device=device, dtype=dtype)
        t = torch.rand((batch_size, 2, 1), device=device, dtype=dtype)
        ss = So2.exp(theta[..., None])
        p1 = ss * t
        p2 = ss.matrix() @ t
        self.assert_close(p1, p2)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_inverse(self, device, dtype, batch_size):
        data = torch.rand(batch_size, 2, device=device, dtype=dtype)
        z = torch.complex(data[..., 0, None], data[..., 1, None])
        s = So2(z)
        self.assert_close(s.inverse().inverse().z.real, z.real)
        self.assert_close(s.inverse().inverse().z.imag, z.imag)
