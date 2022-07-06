import torch

from kornia.geometry.nerf.nerf_model import MLP


class TestNerfModel:
    def test_mlp(self, device, dtype):
        d_input = 63  # Input dimension after encoding
        num_hidden = 256
        mlp = MLP(d_input, num_units=2, num_unit_layers=4, num_hidden=num_hidden)

        num_rays = 15
        num_ray_points = 11
        x = torch.rand(num_rays, num_ray_points, d_input)
        xout = mlp(x)
        assert xout.shape == (num_rays, num_ray_points, num_hidden)
