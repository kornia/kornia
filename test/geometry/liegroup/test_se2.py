import pytest
import torch

from kornia.geometry.liegroup import So2, Se2
from kornia.testing import BaseTester


class TestSe2(BaseTester):
    def test_smoke(self, device, dtype):
        z = torch.rand((2,), dtype=torch.cfloat, device=device)
        so2 = So2(z)
        t = torch.rand((1, 2), device=device, dtype=dtype)
        s = Se2(so2, t)
        assert isinstance(s, Se2)
        assert isinstance(s.r, So2)
        self.assert_close(s.r.z.data, z)
        self.assert_close(s.t, t)

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
            self.assert_close(s1.r.z, z[i][None])
            self.assert_close(s1.t[0], t[i])

    @pytest.mark.parametrize("batch_size", (2, 5))
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
