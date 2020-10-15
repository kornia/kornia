import pytest
import kornia.testing as utils  # test utils

from torch.testing import assert_allclose
from torch.autograd import gradcheck
from kornia.feature.mkd import *


@pytest.mark.parametrize("ps", [5, 13, 25])
def test_get_grid_dict(ps):
    grid_dict = get_grid_dict(ps)
    param_keys = ['x','y','phi','rho']
    assert set(grid_dict.keys()) == set(param_keys)
    for k in param_keys:
        assert grid_dict[k].shape == (ps, ps)


@pytest.mark.parametrize("d1,d2",
                         [(1, 1), (1,2), (2,1), (5,6)])
def test_get_kron_order(d1,d2):
    out = get_kron_order(d1, d2)
    assert out.shape == (d1 * d2, 2)

