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

class TestMKDGradients:
    @pytest.mark.parametrize("ps", [5, 13, 25])
    def test_shape(self, ps, device):
        inp = torch.ones(1, 1, ps, ps, device=device)
        gradients = MKDGradients().to(device)
        out = gradients(inp)
        assert out.shape == (1, 2, ps, ps)

    @pytest.mark.parametrize("bs", [1, 5, 13])
    def test_batch_shape(self, bs, device):
        inp = torch.ones(bs, 1, 15, 15, device=device)
        gradients = MKDGradients().to(device)
        out = gradients(inp)
        assert out.shape == (bs, 2, 15, 15)

    def test_print(self, device):
        gradients = MKDGradients().to(device)
        gradients.__repr__()

    def test_toy(self, device):
        patch = torch.ones(1, 1, 6, 6, device=device).float()
        patch[0, 0, :, 3:] = 0
        gradients = MKDGradients().to(device)
        out = gradients(patch)
        expected = torch.Tensor([0, 0, 1., 1., 0, 0], device=device)
        expected_mags = expected.unsqueeze(0).repeat(6,1)
        expected_oris = expected_mags * 0
        assert_allclose(out[0,0,:,:], expected_mags, atol=1e-3, rtol=1e-3)
        assert_allclose(out[0,1,:,:], expected_oris, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 13, 13
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def grad_describe(patches):
            return MKDGradients()(patches)
        assert gradcheck(grad_describe, (patches),
                         raise_exception=True, nondet_tol=1e-4)
