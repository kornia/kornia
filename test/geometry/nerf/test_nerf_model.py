import torch

from kornia.geometry.nerf.nerf_model import MLP, NerfModel


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

    def test_nerf(self, device, dtype):
        num_ray_points = 11
        nerf_model = NerfModel(
            num_ray_points=num_ray_points,
            num_pos_freqs=10,
            num_dir_freqs=4,
            num_units=2,
            num_nuit_layers=4,
            num_hidden=256,
        )
        num_rays = 15
        origins = torch.rand(num_rays, 3)
        directions = torch.rand(num_rays, 3)
        sigma, rgb = nerf_model(origins, directions)
        assert sigma.shape == (num_rays, num_ray_points, 1)
        assert rgb.shape == (num_rays, num_ray_points, 3)
