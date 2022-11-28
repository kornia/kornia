import pytest
import torch

from kornia.geometry.liegroup import So2, Se2
from kornia.testing import BaseTester


class TestSe2(BaseTester):
    def test_smoke(self, device, dtype):
        z = torch.rand((2,), dtype=torch.cfloat, device=device)
        so2 = So2(z)
        t = torch.rand(1, 2, device=device, dtype=dtype)
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

