import torch

from kornia.geometry.nerf.nerf_model import MLP


class TestNerfModel:
    def test_mlp(self, device, dtype):
        n_freqs = 4
        d_input = 3
        d_enocded = d_input * (2 * n_freqs + 1)
        n_hidden = 256
        mlp = MLP(d_enocded, n_units=2, n_unit_layers=4, n_hidden=n_hidden)

        n_rays = 13
        x = torch.rand(n_rays, d_enocded)
        xout = mlp(x)
        assert xout.shape == (n_rays, n_hidden)
