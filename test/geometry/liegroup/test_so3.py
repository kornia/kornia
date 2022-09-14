import pytest
import torch

from kornia.geometry.liegroup.so3 import So3
from kornia.geometry.quaternion import Quaternion

from kornia.testing import assert_close


class TestSo3:
    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size)
        q1 = q1.to(device, dtype)
        s = So3(q1)
        a = torch.rand(batch_size, 3, device=device, dtype=dtype)
        b = s.exp(a).log()
        assert_close(b,a)
